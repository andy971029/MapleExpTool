"""瀏覽器版的膠水層：JS 把畫面餵進來，這裡用桌面版同一套辨識與追蹤邏輯算。

整個檔案在 Pyodide 裡執行。刻意不碰 ``core/session.py``：那一層綁了地圖面板、
SQLite 與 Windows 視窗管理。經驗值（模板比對）已經在真實畫面上驗證過，現在加上
等級、職業、角色名；地圖與儲存還沒搬。

身分辨識的 OCR 在 JS 端（Tesseract.js），而 Python 這邊的 tick 是同步的。兩邊的
銜接是「掛號制」：Python 把要辨識的圖前處理好、以 PNG data URL 放進 payload 的
``ocr`` 欄位，JS 非同步辨識完再呼叫 ``ocr_result()`` 交回文字。辨識期間 tick 照常
跑，不會卡住經驗值的取樣。

跟桌面版的對應：

* ``FrameCapturer`` 取代 ``win32.capture.WindowCapturer``。介面跟 ``tests/test_autosetup.py``
  裡的假擷取器一樣，所以 ``StatusReader`` 與 ``auto_configure`` 完全不用改。
* ``WebSession.tick()`` 是 ``TrackingSession.tick()`` 的縮水版：只有 reader → tracker。
  驅動者是 JS 的 ``setInterval``，跟桌面版「浮窗用 Tk after、主控台用 while」是
  同一個想法 —— tick 不自己開迴圈。
"""

from __future__ import annotations

import base64
import json
import time

import numpy as np

from mapleexp.autosetup import auto_configure, builtin_templates_path
from mapleexp.config import Config, Roi
from mapleexp.core import jobvocab
from mapleexp.core.stats import format_elapsed, format_exp, format_rate
from mapleexp.core.tracker import ACTIVE, IDLE, PAUSED, WARMUP, Tracker
from mapleexp.vision import identity as ident
from mapleexp.vision import ocr, pngio
from mapleexp.vision.locate import locate_exp_field
from mapleexp.vision.reader import StatusReader
from mapleexp.vision.templates import TemplateSet
from mapleexp.win32.capture import CaptureError, Frame

STATE_LABELS = {
    WARMUP: "等待第一筆經驗",
    ACTIVE: "追蹤中",
    IDLE: "閒置中",
    PAUSED: "無法讀取",
}

# 連續讀不到幾格才懷疑 ROI 偏了（視窗被縮放）並重新定位；兩次重新定位至少隔幾秒。
RELOCATE_AFTER_MISSES = 5
RELOCATE_INTERVAL_SEC = 10.0
WIDEN_CHECK_SEC = 10.0

# 速率與「最近累計」用同一個時間窗。注意它是**活躍時間**窗（閒置時鐘會停），
# 所以掛機或發呆的那幾分鐘不會把速率拉低；整場平均也是除以活躍時間，跟桌面版一致。
RATE_WINDOW_SEC = 600


# 角色名與職業多久重新確認一次（跟桌面版 CHARACTER_RECHECK_SEC 同一個理由：
# 只有那塊像素真的變了才重辨識）。
CHARACTER_RECHECK_SEC = 5.0
# 中文兩行各用這幾個倍率辨識、取字數最多的（桌面版 recognize_best 同一招）。
CHARACTER_OCR_SCALES = (3, 4, 6)
# 等級方塊只有 22x13，要多個倍率互相印證。實測 Tesseract 的英文模型在 3/4/6/8 倍
# 都讀出 45，沒有分歧；中文模型讀同一塊則 4 個倍率只有 2 個有回答，所以數字走英文模型。
LEVEL_OCR_SCALES = (3, 4, 6, 8)
# 同一組字形辨識失敗幾次就放棄（字形變了會重新計算）。
OCR_RETRY_LIMIT = 3


def _png_data_url(gray: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(pngio.encode_png(gray)).decode("ascii")


class FrameCapturer:
    """桌面版 ``WindowCapturer`` 的替身：畫面由 JS 餵進來，這裡只負責切 ROI。"""

    def __init__(self) -> None:
        self.frame: np.ndarray | None = None

    def set_frame(self, rgba, width: int, height: int) -> None:
        """收下 canvas 的 ``getImageData`` 結果。

        canvas 給的是 RGBA，桌面版擷取層給的是 BGRA（Windows DIB 的順序），整條
        辨識管線都照 BGRA 寫 —— 這裡換通道，下游就一行都不用改。
        """
        if not isinstance(rgba, (bytes, bytearray, memoryview)):
            rgba = rgba.to_bytes()      # Pyodide 的 JsProxy（Uint8ClampedArray）
        pixels = np.frombuffer(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)
        self.frame = pixels[..., [2, 1, 0, 3]]

    def capture_client(self) -> Frame:
        if self.frame is None:
            raise CaptureError("尚未收到畫面")
        return Frame(pixels=self.frame, backend="browser")

    def capture_roi(self, roi: Roi) -> Frame:
        frame = self.capture_client()
        height, width = frame.pixels.shape[:2]
        left, top, right, bottom = roi.to_pixels(width, height)
        return Frame(pixels=frame.pixels[top:bottom, left:right], backend="browser")

    def close(self) -> None:
        self.frame = None


class WebSession:
    """一場追蹤。JS 每秒呼叫一次 ``tick()``，拿回 JSON 字串畫到頁面上。"""

    def __init__(self, templates: TemplateSet | None = None) -> None:
        # 不讀磁碟上的設定檔：瀏覽器裡沒有 %LOCALAPPDATA%，而且 ROI 每次分享視窗
        # 都重新找，便宜（不到 0.2 秒）又不會沿用到錯的。
        self.cfg = Config()
        self.cfg.tracker.rate_windows = [RATE_WINDOW_SEC]
        self.cfg.tracker.eta_window = RATE_WINDOW_SEC
        self.templates = templates or TemplateSet.load(builtin_templates_path())
        self.capturer = FrameCapturer()
        self.reader: StatusReader | None = None
        self.tracker = Tracker(self.cfg.tracker)
        self.message = "尚未開始"
        self.rect: tuple[int, int, int, int] | None = None
        self._relocated_at = 0.0
        self._widen_checked_at = 0.0

        # 身分：等級、職業、角色名
        self.level_reader = ident.LevelReader()
        self.level: int | None = None
        self.job = ""
        self.character = ""
        self._level_misses: dict[str, int] = {}
        self._character_key = ""
        self._character_checked_at = -CHARACTER_RECHECK_SEC
        self._ocr_outbox: list[dict] = []
        self._ocr_inflight: dict[str, dict] = {}
        self._ocr_ready = False
        # 每一項資料「從畫面哪一塊讀來的」，頁面拿去裁放大圖，讓人肉眼核對辨識有沒有看對。
        self._rois: dict[str, tuple[int, int, int, int] | None] = {"level": None, "job": None, "name": None}
        self._rois_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------ #

    def feed_frame(self, rgba, width: int, height: int) -> None:
        self.capturer.set_frame(rgba, width, height)

    def set_ocr_ready(self, ready: bool) -> None:
        """JS 端的 OCR 引擎載好了才開始掛號；沒載好就掛的號沒人處理，只會卡在待辦裡。"""
        self._ocr_ready = bool(ready)

    def reset(self) -> None:
        """重新計算，但 ROI 留著（視窗沒動就不用重找）。"""
        self.tracker.reset()

    def tick(self) -> str:
        if self.reader is None:
            self._locate(force=True)
            if self.reader is None:
                return self._payload(None)

        try:
            reading = self.reader.read()
        except CaptureError as exc:
            self.message = str(exc)
            return self._payload(None)

        self.tracker.feed(reading)
        snapshot = self.tracker.snapshot()
        if reading.ok:
            self._scan_identity()
            self._maybe_widen_roi()

        if not reading.ok:
            self.message = reading.reason
            # 連續讀不到可能是視窗被縮放、ROI 偏了。跟桌面版 auto_configure 一樣：
            # 先試讀舊 ROI，讀不到才重新定位，定位也失敗就沿用舊的等它回來。
            if snapshot.consecutive_misses >= RELOCATE_AFTER_MISSES:
                now = time.monotonic()
                if now - self._relocated_at >= RELOCATE_INTERVAL_SEC:
                    self._relocated_at = now
                    self._locate(force=False)
        else:
            self.message = ""
        return self._payload(snapshot, reading.raw_exp)

    # ------------------------------------------------------------------ #

    def _locate(self, force: bool) -> None:
        auto = auto_configure(
            self.cfg, self.capturer, self.templates, force=force, template_source="內建模板"
        )
        self.message = auto.message
        if auto.located is not None:
            self.rect = auto.located.rect
        if auto.ok and self.reader is None:
            self.reader = StatusReader(self.capturer, self.cfg.reader, self.templates)

    def _maybe_widen_roi(self) -> None:
        """欄位被滑鼠或特效遮住時重新定位，只會找到露出來的那一截，ROI 就此變窄、
        遮蔽移開後也不會自己長回來（實測 68 寬的欄位縮成 50，開頭數字被切掉）。
        所以讀得到時也定期再找一次，只在找到「更寬」且讀得到百分比的結果才採用；
        被遮住的那幾格只會找到更窄的，不會誤蓋掉好的 ROI。
        """
        now = time.monotonic()
        if now - self._widen_checked_at < WIDEN_CHECK_SEC:
            return
        self._widen_checked_at = now
        frame = self.capturer.frame
        roi = self.cfg.reader.exp_roi
        if frame is None or not roi.is_set():
            return
        found = locate_exp_field(frame, self.templates)
        if found is None or not found.confident:
            return
        left, _top, right, _bottom = self._current_rect() or (0, 0, 0, 0)
        if found.rect[2] - found.rect[0] <= right - left:
            return
        self.cfg.reader.exp_roi = found.roi
        self.cfg.reader.threshold = found.threshold
        self.cfg.reader.invert = found.invert
        self.cfg.reader.ink_color = None
        self.rect = found.rect

    # ---------------------------------------------------------------- 身分

    def _scan_identity(self) -> None:
        """每格掃等級；角色名只有在像素變了、且距上次夠久才重辨識。

        跟桌面版一樣，等級走「模板優先、認不出才 OCR」：OCR 的結果會教給模板，
        之後都走快路。
        """
        frame = self.capturer.frame
        rect = self._current_rect()
        if frame is None or rect is None:
            return
        found = ident.scan(frame, None, exp_rect=rect)
        # 版面只有換解析度才會變，所以找到過的位置要記住；某一格沒掃到（色鍵被特效蓋住、
        # 名牌被遮一下）就清掉，頁面會每隔幾秒閃一次「等待定位」。
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if size != self._rois_size:
            self._rois = {"level": None, "job": None, "name": None}
            self._rois_size = size
        job_rect, name_rect = _band_rects(found.name_rect, found.name_image)
        for key, value in (("level", found.level_rect), ("job", job_rect), ("name", name_rect)):
            if value is not None:
                self._rois[key] = value

        zone = found.zone
        if zone is not None and zone.digits:
            value = self.level_reader.read_templates(zone)
            if value is not None:
                self.level = value
            elif self._ocr_ready and self._level_misses.get(zone.shape_key, 0) < OCR_RETRY_LIMIT:
                self._request_level(zone)

        if found.name_image is not None and self._ocr_ready:
            self._maybe_request_character(found.name_image)

    def _request_level(self, zone) -> None:
        images = [_png_data_url(ocr.prepare(zone.image, scale=s)) for s in LEVEL_OCR_SCALES]
        self._request_ocr(f"level:{zone.shape_key}", "digits", images, {"zone": zone})

    def _maybe_request_character(self, image: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._character_checked_at < CHARACTER_RECHECK_SEC:
            return
        self._character_checked_at = now
        key = ident.fingerprint(image)
        if key == self._character_key and self.character:
            return
        bands = ident.text_bands(image)
        if not bands:
            return
        # 職業在上、角色名在下；兩行靠太近時引擎會黏成一行，所以自己切開各辨識。
        images = [
            [_png_data_url(ocr.prepare(image[top:bottom], scale=s)) for s in CHARACTER_OCR_SCALES]
            for top, bottom in bands[:2]
        ]
        self._request_ocr(f"character:{key}", "text", images, {"key": key})

    def _request_ocr(self, rid: str, kind: str, images: list, ctx: dict) -> None:
        if rid in self._ocr_inflight:
            return
        self._ocr_inflight[rid] = ctx
        self._ocr_outbox.append({"id": rid, "kind": kind, "images": images})

    def ocr_result(self, rid: str, texts_json: str | None) -> None:
        """JS 辨識完交回結果。``texts_json`` 是 JSON；None 代表引擎出錯。"""
        ctx = self._ocr_inflight.pop(rid, None)
        if ctx is None:
            return
        texts = json.loads(texts_json) if texts_json else None
        if rid.startswith("level:"):
            self._finish_level(ctx["zone"], texts)
        elif rid.startswith("character:"):
            self._finish_character(ctx, texts)

    def _finish_level(self, zone, texts) -> None:
        """每個倍率各給一個答案，要全部同意才採用（桌面版 read_level_by_ocr 同一個規則）。"""
        answers = set()
        for text in texts or []:
            cleaned = (text or "").translate(ident._DIGIT_LOOKALIKES).strip()
            if not cleaned:
                continue
            if not cleaned.isdigit() or not 1 <= len(cleaned) <= 3 or not 1 <= int(cleaned) <= ident.MAX_LEVEL:
                answers.add(None)       # 讀出不像等級的東西：整次不信
                continue
            answers.add(int(cleaned))
        if len(answers) != 1 or None in answers:
            key = zone.shape_key
            self._level_misses[key] = self._level_misses.get(key, 0) + 1
            return
        value = answers.pop()
        self._level_misses.pop(zone.shape_key, None)
        # 字數對得上才教：對不上代表切字形跟辨識看到的不是同一回事。
        if len(str(value)) == len(zone.digits):
            self.level_reader.teach(zone, value)
        self.level = value

    def _finish_character(self, ctx: dict, texts) -> None:
        """``texts`` 是 [[行1各倍率], [行2各倍率]]，每行取字數最多的。"""
        if not texts:
            return      # 讀不到就保留上一次的結果，不要把已知的名字洗掉
        lines = []
        for candidates in texts:
            best = max((ocr.tidy(c).replace("\n", "") for c in candidates), key=len, default="")
            if best:
                lines.append(best)
        if not lines:
            return
        if len(lines) >= 2:
            guess = jobvocab.identify(lines[0])
            # 配不上清單時不要蓋掉已經讀對的職業；還沒有職業時才照原文存。
            self.job = guess.name if guess.matched or not self.job else self.job
            self.character = lines[1]
        else:
            self.character = lines[0]
        self._character_key = ctx["key"]

    def _take_ocr_outbox(self) -> list[dict]:
        outbox, self._ocr_outbox = self._ocr_outbox, []
        return outbox

    def _current_rect(self) -> tuple[int, int, int, int] | None:
        """用目前這張畫面的尺寸重算欄位位置。ROI 是距底部中央的偏移，視窗縮放後
        reader 仍讀得到，但定位時記下的像素座標已經過期；拿舊座標去畫放大圖會是空的。"""
        frame = self.capturer.frame
        roi = self.cfg.reader.exp_roi
        if frame is None or not roi.is_set():
            return self.rect
        return roi.to_pixels(int(frame.shape[1]), int(frame.shape[0]))

    def _payload(self, snapshot, raw_exp: str = "") -> str:
        frame = self.capturer.frame
        rect = self._current_rect()
        data = {
            "located": self.reader is not None,
            "message": self.message,
            "rect": list(rect) if rect else None,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
            "raw_exp": raw_exp,
            "level": self.level,
            "job": self.job,
            "character": self.character,
            "ocr": self._take_ocr_outbox(),
            "rois": {"exp": list(rect) if rect else None,
                     **{k: list(v) if v else None for k, v in self._rois.items()}},
        }
        if snapshot is None:
            data["state"] = "等待經驗值欄位"
            return json.dumps(data, ensure_ascii=False)

        recent = snapshot.rates.get(RATE_WINDOW_SEC)
        per_hour = recent.exp_per_hour if recent else None
        average = (
            snapshot.cum_net / (snapshot.active_sec / 3600.0) if snapshot.active_sec > 0 else None
        )
        stats = {
            "per_hour": format_rate(per_hour),
            "per_half_hour": format_exp(per_hour / 2 if per_hour is not None else None),
            "total": format_exp(snapshot.cum_net),
            "average": format_rate(average),
            "recent": format_exp(recent.exp_gained if recent and recent.valid else None),
            "recent_valid": bool(recent and recent.valid),
            "window_text": _window_label(RATE_WINDOW_SEC),
            "span_text": format_elapsed(recent.span_sec) if recent and recent.valid else "--",
        }
        data.update({
            "state": STATE_LABELS.get(snapshot.state, snapshot.state),
            "state_key": snapshot.state,
            "exp_abs": snapshot.exp_abs,
            "exp_pct": snapshot.exp_pct,
            "need": snapshot.need,
            "remaining": snapshot.remaining,
            "exp_text": format_exp(snapshot.exp_abs),
            "need_text": format_exp(snapshot.need),
            "active_text": format_elapsed(snapshot.active_sec),
            "idle_text": format_elapsed(snapshot.idle_sec),
            "eta_text": _format_eta(snapshot.eta_sec),
            "eta_window": _window_label(snapshot.eta_window) if snapshot.eta_sec else "",
            "stats": stats,
            "samples": snapshot.samples,
            "misses": snapshot.misses,
            "consecutive_misses": snapshot.consecutive_misses,
            "levelups": snapshot.levelups,
            "deaths": snapshot.deaths,
        })
        return json.dumps(data, ensure_ascii=False)


def _band_rects(name_rect, name_image):
    """名牌區塊切出的兩行（職業在上、角色名在下）各自換算成畫面座標。

    OCR 也是照同樣的切法辨識，所以頁面上看到的裁圖就是引擎真正看到的東西。
    """
    if name_rect is None or name_image is None:
        return None, None
    bands = ident.text_bands(name_image)
    left, top, right, _bottom = name_rect
    rects = [(left, top + a, right, top + b) for a, b in bands[:2]]
    if len(rects) < 2:
        return None, (rects[0] if rects else None)
    return rects[0], rects[1]


def _window_label(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600} 小時"
    return f"{seconds // 60} 分鐘"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"
