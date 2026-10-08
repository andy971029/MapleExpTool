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
from mapleexp.gamedata import MapRecord, MapVocabulary
from mapleexp.core import jobvocab, mapvocab
from mapleexp.core.stats import format_elapsed, format_exp, format_rate
from mapleexp.core.tracker import ACTIVE, IDLE, PAUSED, WARMUP, Tracker
from mapleexp.vision import identity as ident
from mapleexp.vision import ocr, panels, pngio
from mapleexp.core.reading import StatusReading
from mapleexp.vision.locate import closing_bracket_columns, locate_exp_field
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
# 小地圖面板要靠標題列「小地圖」三個字認（桌面版 PanelTracker 同一招），OCR 是
# 非同步的，所以搜尋失敗後隔幾秒再試；搜尋範圍沿用桌面版（畫面左上角的比例）。
MINIMAP_TITLE = "小地圖"
MINIMAP_SEARCH_INTERVAL_SEC = 5.0
MINIMAP_TITLE_SCALES = (3, 4, 5)
MINIMAP_MAX_BARS = 6
# 面板還記著、卻連續這麼多格讀不到地圖名區，才當作面板被搬走或收合而重找。
MAP_LOST_TICKS = 5
MAP_NAME_RETRY_SEC = 5.0
MAP_OCR_SCALES = (3, 4, 6)
# 同一組字形辨識失敗幾次就放棄（字形變了會重新計算）。
OCR_RETRY_LIMIT = 3
LEVEL_SLOW_RETRY_SEC = 10.0


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


EMPTY_ROIS: dict[str, tuple[int, int, int, int] | None] = {
    "level": None, "job": None, "name": None, "map": None,
}


class WebSession:
    """一場追蹤。JS 每秒呼叫一次 ``tick()``，拿回 JSON 字串畫到頁面上。"""

    def __init__(self, templates: TemplateSet | None = None, require_bracket: bool = True) -> None:
        # 綠色的 ``]`` 看不到就代表欄位被擋住，這一格不計。合成畫面沒有綠色括號，
        # 單元測試用 False 跳過；真的遊戲畫面一律要開。
        self.require_bracket = require_bracket
        self.occluded = False
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
        self._level_last_try = 0.0
        self.boost = 1.0
        self.level_status = ""
        self._character_key = ""
        self._character_checked_at = -CHARACTER_RECHECK_SEC
        self._ocr_outbox: list[dict] = []
        self._ocr_inflight: dict[str, dict] = {}
        self._ocr_ready = False
        # 每一項資料「從畫面哪一塊讀來的」，頁面拿去裁放大圖，讓人肉眼核對辨識有沒有看對。
        self._rois: dict[str, tuple[int, int, int, int] | None] = dict(EMPTY_ROIS)
        self._rois_size: tuple[int, int] | None = None

        # 地圖：先找到小地圖面板，再從面板內容區定位名稱區；指紋連續穩定才算換圖。
        self.map_watcher = ident.MapWatcher()
        self.map_id = ""
        self.map_name = ""
        self._map_names: dict[str, str] = {}
        self._map_scores: dict[str, float] = {}
        self.map_vocab = MapVocabulary()
        self._minimap_body: tuple[int, int, int, int] | None = None
        self._minimap_searched_at = -MINIMAP_SEARCH_INTERVAL_SEC
        # 給畫面顯示「為什麼還沒找到小地圖」：卡在哪一步一眼就看得出來。
        self.map_status = "等待文字辨識引擎"
        self._map_misses = 0
        self._map_name_checked_at = -MAP_NAME_RETRY_SEC

    # ------------------------------------------------------------------ #

    def feed_frame(self, rgba, width: int, height: int) -> None:
        self.capturer.set_frame(rgba, width, height)

    def set_boost(self, factor: float) -> None:
        """經驗加倍券試算：只放大「收益」類的顯示數字，剩餘經驗與存檔用的原始累計不動。"""
        self.boost = max(1.0, float(factor))

    def set_ocr_ready(self, ready: bool) -> None:
        """JS 端的 OCR 引擎載好了才開始掛號；沒載好就掛的號沒人處理，只會卡在待辦裡。"""
        self._ocr_ready = bool(ready)

    def set_map_vocab(self, rows_json: str) -> int:
        """載入網頁附的地圖清單快照（``[[id, street, name], ...]``），回傳筆數。"""
        rows = json.loads(rows_json)
        self.map_vocab = MapVocabulary(
            tuple(MapRecord(str(i), str(street), str(name)) for i, street, name in rows),
            source="快照",
        )
        return len(self.map_vocab)

    def resume(self) -> None:
        """暫停／重新分享後呼叫：下一筆讀數只當新基準，空窗期間的經驗與時間都不算。"""
        self.tracker.mark_discontinuity()

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

        reading = self._reject_if_occluded(reading)
        self.tracker.feed(reading)
        snapshot = self.tracker.snapshot()
        if reading.ok:
            self._scan_identity()
            self._maybe_widen_roi()

        if not reading.ok:
            self.message = reading.reason
            # 連續讀不到可能是視窗被縮放、ROI 偏了。跟桌面版 auto_configure 一樣：
            # 先試讀舊 ROI，讀不到才重新定位，定位也失敗就沿用舊的等它回來。
            # 被擋住時重新定位只會抓到殘缺的一截，等它露出來比較好。
            if snapshot.consecutive_misses >= RELOCATE_AFTER_MISSES and not self.occluded:
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

    def _reject_if_occluded(self, reading: StatusReading) -> StatusReading:
        """欄位右端的綠色 ``]`` 不見了：資料不齊全，寧可這格不算，也不要把讀到一半的數字記進去。"""
        self.occluded = False
        rect = self._current_rect()
        frame = self.capturer.frame
        if not self.require_bracket or not reading.ok or rect is None or frame is None:
            return reading
        if closing_bracket_columns(frame, rect):
            return reading
        self.occluded = True
        return StatusReading.failed(
            reading.mono, reading.wall,
            reason="經驗值欄位被擋住（看不到綠色的 ]），資料不完整，暫停計算",
            raw_exp=reading.raw_exp,
        )

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
        found = ident.scan(frame, None, exp_rect=rect, map_rect=self._minimap_body)
        # 版面只有換解析度才會變，所以找到過的位置要記住；某一格沒掃到（色鍵被特效蓋住、
        # 名牌被遮一下）就清掉，頁面會每隔幾秒閃一次「等待定位」。
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if size != self._rois_size:
            self._rois = dict(EMPTY_ROIS)
            self._rois_size = size
            self._forget_minimap()
            self.map_watcher = ident.MapWatcher()
            self.map_id = self.map_name = ""
        job_rect, name_rect = _band_rects(found.name_rect, found.name_image)
        for key, value in (("level", found.level_rect), ("job", job_rect), ("name", name_rect)):
            if value is not None:
                self._rois[key] = value

        zone = found.zone
        if zone is not None and zone.digits:
            value = self.level_reader.read_templates(zone)
            if value is not None:
                self.level = value
            elif self._ocr_ready and self._level_may_retry(zone.shape_key):
                self._request_level(zone)

        if found.name_image is not None and self._ocr_ready:
            self._maybe_request_character(found.name_image)

        self._scan_map(frame, found)

    def _forget_minimap(self) -> None:
        self._minimap_body = None
        self._rois["map"] = None
        self._map_misses = 0

    def _scan_map(self, frame: np.ndarray, found) -> None:
        if self._minimap_body is None:
            if self._ocr_ready:
                self._maybe_search_minimap(frame)
            else:
                self.map_status = "等待文字辨識引擎載入"
            return
        if not found.map_id:
            # 面板還記著卻讀不到名稱區：連續幾格才算面板不見了（換圖的讀取畫面也會短暫讀不到）。
            self._map_misses += 1
            if self._map_misses >= MAP_LOST_TICKS:
                self._forget_minimap()
            return
        self._map_misses = 0
        self._rois["map"] = found.map_rect
        if self.map_watcher.feed(found.map_id, found.map_image):
            self.map_id = self.map_watcher.map_id
            self.map_name = self._map_names.get(self.map_id, "")
        # 有清單時，沒讀到滿分就隔一陣子重讀（OCR 在這個字級下每次結果會跳動，換圖當下
        # 剛好讀壞名字就會一路錯下去；桌面版 _maybe_refresh_map_name 同一個理由）。
        if self.map_id and self._ocr_ready and self._map_scores.get(self.map_id, -1.0) < (
            1.0 if self.map_vocab else 0.0
        ):
            self._maybe_request_map_name()

    def _maybe_search_minimap(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._minimap_searched_at < MINIMAP_SEARCH_INTERVAL_SEC:
            return
        if any(key.startswith("title:") for key in self._ocr_inflight):
            return
        self._minimap_searched_at = now
        height, width = frame.shape[:2]
        region = frame[: int(height * ident.MINIMAP_SEARCH_H), : int(width * ident.MINIMAP_SEARCH_W)]
        bars = [rect for rect, _fill in panels.find_title_bars(region)][:MINIMAP_MAX_BARS]
        if not bars:
            self.map_status = "畫面左上角找不到白底標題列：小地圖是否已展開、有沒有被遮住？"
            return
        self.map_status = f"找到 {len(bars)} 條標題列，辨識文字中…"
        images = []
        for left, top, right, bottom in bars:
            crop = frame[max(0, top - 2) : bottom + 2, max(0, left - 2) : right + 2]
            images.append([_png_data_url(ocr.prepare(crop, scale=s, bright_text=False))
                           for s in MINIMAP_TITLE_SCALES])
        self._request_ocr("title:minimap", "text", images,
                          {"bars": bars, "size": (int(width), int(height))})

    def _finish_minimap_search(self, ctx: dict, texts) -> None:
        """哪一條標題列讀起來像「小地圖」，它底下就是名稱區所在的內容區。"""
        frame = self.capturer.frame
        if not texts or frame is None:
            self.map_status = "標題列的文字辨識沒有回答"
            return
        height, width = frame.shape[:2]
        if ctx["size"] != (int(width), int(height)):
            return      # 辨識途中換了解析度，座標已經作廢
        for (left, top, right, bottom), candidates in zip(ctx["bars"], texts):
            for text in candidates:
                # Tesseract 的 chi_tra 會在字與字之間補空白、把框線讀成「|」（實測 "| 小 地 圖 |"），
                # 只留文字本身才配得上。
                title, _score = panels.match_title("".join(c for c in (text or "") if c.isalnum()))
                if title == MINIMAP_TITLE:
                    self._minimap_body = (
                        max(0, left - panels.BODY_PAD_LEFT),
                        bottom,
                        min(width, right + panels.BODY_PAD_RIGHT),
                        min(height, bottom + panels.BODY_HEIGHT),
                    )
                    self._map_misses = 0
                    self.map_status = ""
                    return
        read = [(c[0] if c else "") for c in texts]
        self.map_status = "標題列讀到「" + "」「".join(t.replace("\n", "")[:8] for t in read) + "」，都不是「小地圖」"

    def _maybe_request_map_name(self) -> None:
        image = self.map_watcher.image
        now = time.monotonic()
        if image is None or now - self._map_name_checked_at < MAP_NAME_RETRY_SEC:
            return
        if any(key.startswith("map:") for key in self._ocr_inflight):
            return
        bands = ident.text_bands(image)
        if not bands:
            return
        self._map_name_checked_at = now
        # 名稱區通常是兩行（地區／地圖名），分行辨識跟角色名一樣比較穩。
        images = [
            [_png_data_url(ocr.prepare(image[top:bottom], scale=s)) for s in MAP_OCR_SCALES]
            for top, bottom in bands[:2]
        ]
        bars = trailing_numeral_bars(image, bands[min(1, len(bands) - 1)])
        self._request_ocr(
            f"map:{self.map_id}", "text", images, {"map_id": self.map_id, "bars": bars}
        )

    def _finish_map_name(self, ctx: dict, texts) -> None:
        lines = []
        for candidates in texts or []:
            best = max((ocr.tidy(c).replace("\n", "") for c in candidates), key=len, default="")
            if best:
                lines.append(best)
        if not lines:
            return
        map_id = ctx["map_id"]
        if self.map_vocab:
            # 一定挑出最接近的一張，沒有門檻：真正區分地圖的是像素指紋，名字錯了只是顯示錯。
            guess = mapvocab.identify("".join(lines), self.map_vocab)
            if not guess.full_name or guess.score <= self._map_scores.get(map_id, -1.0):
                return      # 這次沒有比上次好，保留原本的
            name, score = guess.full_name, guess.score
            fixed = _fix_numeral(guess, ctx.get("bars", 0), self.map_vocab)
            if fixed:
                name, score = fixed, 1.0
        else:
            name, score = " · ".join(lines), 0.0
        self._map_names[map_id] = name
        self._map_scores[map_id] = score
        if map_id == self.map_id:
            self.map_name = name

    def _request_level(self, zone) -> None:
        images = [_png_data_url(ocr.prepare(zone.image, scale=s, contrast=True)) for s in LEVEL_OCR_SCALES]
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
        elif rid.startswith("title:"):
            self._finish_minimap_search(ctx, texts)
        elif rid.startswith("map:"):
            self._finish_map_name(ctx, texts)

    def _level_may_retry(self, key: str) -> bool:
        """連續讀不出來幾次之後改成慢慢重試，而不是永遠放棄。

        永遠放棄的話，剛開頁面那幾格（串流還在編碼、畫面糊）失敗三次，等級就一輩子
        顯示 --，直到字形剛好變了才再試。
        """
        if self._level_misses.get(key, 0) < OCR_RETRY_LIMIT:
            return True
        return time.monotonic() - self._level_last_try >= LEVEL_SLOW_RETRY_SEC

    def _finish_level(self, zone, texts) -> None:
        """每個倍率各給一個答案，同樣位數的答案要至少兩個、且全部一致才採用。

        位數用 ``zone.digits``（色鍵切出來的字形數）當基準：Tesseract 在糊掉的畫面上常
        只回答半個數字（實測 59 在 8 倍讀成 9），這種答案位數不對，直接當作沒回答，
        而不是讓它否決其他倍率的正確答案。位數對卻互相矛盾的才算分歧，整次不信。
        """
        self._level_last_try = time.monotonic()
        expected = len(zone.digits)
        raw = [(t or "").translate(ident._DIGIT_LOOKALIKES).strip() for t in texts or []]
        valid = [int(t) for t in raw if t.isdigit() and 1 <= int(t) <= ident.MAX_LEVEL]
        exact = [v for v in valid if len(str(v)) == expected]
        self.level_status = "OCR：" + " / ".join(t or "空" for t in raw)
        # 位數對的答案優先；一個都沒有時，全體一致的答案仍採用但不拿來教模板
        # （字形切分很可能是錯的，教進去會記住錯字形）。
        answers = exact or valid
        if len(answers) < 2 or len(set(answers)) != 1:
            key = zone.shape_key
            self._level_misses[key] = self._level_misses.get(key, 0) + 1
            return
        value = answers[0]
        self._level_misses.pop(zone.shape_key, None)
        self.level_status = ""
        if exact:
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
            "map_name": self.map_name,
            "map_id": self.map_id,
            "level_status": self.level_status if self.level is None else "",
            "map_status": self.map_status if self._minimap_body is None else "",
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
        k = self.boost
        scaled = lambda v: v * k if v is not None else None  # noqa: E731
        stats = {
            "per_hour": format_rate(scaled(per_hour)),
            "per_half_hour": format_exp(per_hour * k / 2 if per_hour is not None else None),
            "total": format_exp(snapshot.cum_net * k),
            "average": format_rate(scaled(average)),
            "recent": format_exp(scaled(recent.exp_gained) if recent and recent.valid else None),
            # 換算成「目前等級所需經驗」的百分比，比純數字直觀；需要經驗量未知時為 "--"。
            "pct": {
                "per_hour": _pct_of(scaled(per_hour), snapshot.need, "/hr"),
                "per_half_hour": _pct_of(per_hour * k / 2 if per_hour is not None else None, snapshot.need),
                "average": _pct_of(scaled(average), snapshot.need, "/hr"),
                "recent": _pct_of(scaled(recent.exp_gained) if recent and recent.valid else None, snapshot.need),
                "total": _pct_of(snapshot.cum_net * k, snapshot.need),
            },
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
            "eta_text": _format_eta(snapshot.eta_sec / k if snapshot.eta_sec is not None else None),
            "eta_window": _window_label(snapshot.eta_window) if snapshot.eta_sec else "",
            "stats": stats,
            # 存檔用的原始數字；畫面上的字串格式化過，不能拿來存。
            "totals": {
                "cum_net": snapshot.cum_net,
                "active_sec": snapshot.active_sec,
                "idle_sec": snapshot.idle_sec,
                "average": average,
            },
            "samples": snapshot.samples,
            "misses": snapshot.misses,
            "consecutive_misses": snapshot.consecutive_misses,
            "levelups": snapshot.levelups,
            "deaths": snapshot.deaths,
        })
        return json.dumps(data, ensure_ascii=False)


def trailing_numeral_bars(image: np.ndarray, band: tuple[int, int]) -> int:
    """數最後一行右端的羅馬數字有幾根直槓（Ⅰ=1、Ⅱ=2、Ⅲ=3），認不出回傳 0。

    Tesseract 會把「Ⅲ」讀成單一個「I」（實測 戰火之地 沼澤地Ⅲ → 沼澤地I），而「沼澤地Ⅰ」
    本身就是清單裡的合法地圖，字串比對無從分辨；但直槓在像素上等距（這個字型是 4 px），
    數得出來。等距這個條件是為了不把前一個漢字的豎筆（距離 3 px）算成多一根。
    """
    top, bottom = band
    luma = ident.to_luma(image)
    mask = (luma >= max(120, int(luma.mean()) + 20))[top:bottom]
    if mask.size == 0 or not mask.any():
        return 0
    counts = mask.sum(axis=0)
    tall = counts >= max(3, int(mask.shape[0] * 0.65))
    last_ink = int(np.flatnonzero(mask.any(axis=0))[-1])
    # 抓出每一道「又高又細」的直槓，記錄起點；最右邊那道必須貼著最後一個有字的欄。
    starts: list[int] = []
    x = last_ink
    while x >= 0:
        if tall[x]:
            end = x
            while x >= 0 and tall[x]:
                x -= 1
            if end - x > 2:
                break
            starts.append(x + 1)
        else:
            x -= 1
            if starts and starts[-1] - x > 5:
                break
    if not starts or last_ink - starts[0] > 2:
        return 0
    bars = 1
    for newer, older in zip(starts, starts[1:]):
        if newer - older != 4:
            break
        bars += 1
    return min(bars, 3)


def _fix_numeral(guess, bars: int, vocab):
    """用像素數出的直槓數，在同一個名稱的 Ⅰ/Ⅱ/Ⅲ 之間改選；沒有對應的就不動。"""
    if not 1 <= bars <= 3:
        return None
    name = mapvocab.clean(guess.name)
    stem = name.rstrip("I")
    have = len(name) - len(stem)
    if not 1 <= have <= 3 or have == bars:
        return None
    street, target = mapvocab.clean(guess.street), stem + "I" * bars
    for record in vocab.records:
        if mapvocab.clean(record.street) == street and mapvocab.clean(record.name) == target:
            return f"{record.street} {record.name}".strip()
    return None


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
        return f"{seconds // 3600} hr"
    return f"{seconds // 60} min"


def _pct_of(value: float | None, need: int | None, suffix: str = "") -> str:
    if value is None or not need:
        return "--"
    return f"{value / need * 100:.2f}%{suffix}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"
