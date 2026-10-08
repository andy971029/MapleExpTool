"""瀏覽器版的膠水層：JS 把畫面餵進來，這裡用桌面版同一套辨識與追蹤邏輯算。

整個檔案在 Pyodide 裡執行。刻意不碰 ``core/session.py``：那一層綁了身分辨識
（Windows OCR）、地圖面板、SQLite，第一版網頁只驗證「分享視窗拿到的像素，用現有
的模板比對讀不讀得到經驗值」這一件事。讀得到，再把其餘的搬過來；讀不到，整個
方向要重新評估，後面的東西也不用做了。

跟桌面版的對應：

* ``FrameCapturer`` 取代 ``win32.capture.WindowCapturer``。介面跟 ``tests/test_autosetup.py``
  裡的假擷取器一樣，所以 ``StatusReader`` 與 ``auto_configure`` 完全不用改。
* ``WebSession.tick()`` 是 ``TrackingSession.tick()`` 的縮水版：只有 reader → tracker。
  驅動者是 JS 的 ``setInterval``，跟桌面版「浮窗用 Tk after、主控台用 while」是
  同一個想法 —— tick 不自己開迴圈。
"""

from __future__ import annotations

import json
import time

import numpy as np

from mapleexp.autosetup import auto_configure, builtin_templates_path
from mapleexp.config import Config, Roi
from mapleexp.core.stats import format_elapsed, format_exp, format_rate
from mapleexp.core.tracker import ACTIVE, IDLE, PAUSED, WARMUP, Tracker
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
        self.templates = templates or TemplateSet.load(builtin_templates_path())
        self.capturer = FrameCapturer()
        self.reader: StatusReader | None = None
        self.tracker = Tracker(self.cfg.tracker)
        self.message = "尚未開始"
        self.rect: tuple[int, int, int, int] | None = None
        self._relocated_at = 0.0

    # ------------------------------------------------------------------ #

    def feed_frame(self, rgba, width: int, height: int) -> None:
        self.capturer.set_frame(rgba, width, height)

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

    def _payload(self, snapshot, raw_exp: str = "") -> str:
        frame = self.capturer.frame
        data = {
            "located": self.reader is not None,
            "message": self.message,
            "rect": list(self.rect) if self.rect else None,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
            "raw_exp": raw_exp,
        }
        if snapshot is None:
            data["state"] = "等待經驗值欄位"
            return json.dumps(data, ensure_ascii=False)

        rates = []
        for window in sorted(snapshot.rates):
            estimate = snapshot.rates[window]
            rates.append({
                "window": window,
                "label": _window_label(window),
                "text": format_rate(estimate.exp_per_hour),
                "valid": estimate.valid,
            })
        data.update({
            "state": STATE_LABELS.get(snapshot.state, snapshot.state),
            "state_key": snapshot.state,
            "exp_abs": snapshot.exp_abs,
            "exp_pct": snapshot.exp_pct,
            "need": snapshot.need,
            "remaining": snapshot.remaining,
            "exp_text": format_exp(snapshot.exp_abs),
            "need_text": format_exp(snapshot.need),
            "gross_text": format_exp(snapshot.cum_gross),
            "active_text": format_elapsed(snapshot.active_sec),
            "idle_text": format_elapsed(snapshot.idle_sec),
            "eta_text": _format_eta(snapshot.eta_sec),
            "eta_window": _window_label(snapshot.eta_window) if snapshot.eta_sec else "",
            "rates": rates,
            "samples": snapshot.samples,
            "misses": snapshot.misses,
            "consecutive_misses": snapshot.consecutive_misses,
            "levelups": snapshot.levelups,
            "deaths": snapshot.deaths,
        })
        return json.dumps(data, ensure_ascii=False)


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
