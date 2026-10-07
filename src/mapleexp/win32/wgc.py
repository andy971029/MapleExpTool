"""Windows Graphics Capture 後端。

這是唯一「不管遊戲是視窗、全螢幕、還是被別的視窗蓋住，都抓得到」的做法：
WGC 要的是視窗自己在 DWM 裡的合成結果，而不是螢幕上看得到的像素。
OBS 的視窗擷取走的也是這條路。

實測這個遊戲（Unity + D3D11）：``PrintWindow`` 直接失敗，從螢幕 BitBlt 則會
在被遮住時抓到別的視窗 —— 只有 WGC 兩種情況都能正確拿到畫面。

需要 ``windows-capture`` 套件（Rust 寫的 abi3 輪子）。沒裝的話擷取層會自動
退回螢幕 BitBlt，功能仍可用，只是要求遊戲畫面沒被蓋住。

WGC 的 API 是「幀到了就回呼」的推送模型，但追蹤器要的是「現在給我最新一張」。
所以這裡用一個背景擷取 + 最新幀插槽把它轉成拉取模型。
"""

from __future__ import annotations

import threading

import numpy as np

from .windows import client_origin_on_screen, get_client_size, window_rect

try:  # pragma: no cover - 取決於環境有沒有裝
    from windows_capture import WindowsCapture

    WGC_AVAILABLE = True
    WGC_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    WindowsCapture = None  # type: ignore[assignment]
    WGC_AVAILABLE = False
    WGC_IMPORT_ERROR = str(exc)


class WgcError(RuntimeError):
    pass


class WgcSession:
    """背景跑一個 WGC 擷取，隨時可以取出最新一幀。"""

    def __init__(self, hwnd: int, min_interval_ms: int = 150) -> None:
        if not WGC_AVAILABLE:
            raise WgcError(f"未安裝 windows-capture（{WGC_IMPORT_ERROR}）")
        self.hwnd = int(hwnd)
        self.min_interval_ms = int(min_interval_ms)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._arrived = threading.Event()
        self._control = None
        self._closed = False
        self._error = ""

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._control is not None:
            return

        capture = WindowsCapture(
            window_hwnd=self.hwnd,
            cursor_capture=False,
            draw_border=False,
            # 我們只需要每秒一兩張，沒必要讓它用螢幕更新率狂送幀。
            minimum_update_interval=self.min_interval_ms,
        )

        @capture.event
        def on_frame_arrived(frame, _control):  # pragma: no cover - 由背景執行緒呼叫
            # frame_buffer 指向對映中的紋理，離開回呼就失效，一定要複製。
            try:
                buffer = np.asarray(frame.frame_buffer)
                with self._lock:
                    self._frame = buffer.copy()
                self._arrived.set()
            except Exception as exc:  # 不要讓背景執行緒的例外悄悄殺掉擷取
                self._error = str(exc)

        @capture.event
        def on_closed():  # pragma: no cover
            self._closed = True

        try:
            self._control = capture.start_free_threaded()
        except Exception as exc:
            raise WgcError(f"WGC 啟動失敗：{exc}") from exc

    def wait_for_first_frame(self, timeout: float = 3.0) -> bool:
        return self._arrived.wait(timeout)

    @property
    def closed(self) -> bool:
        return self._closed

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        control, self._control = self._control, None
        if control is not None:
            try:
                control.stop()
            except Exception:
                pass
        with self._lock:
            self._frame = None
        self._arrived.clear()

    # ------------------------------------------------------------------ #

    def client_frame(self) -> np.ndarray | None:
        """把最新一幀裁成 client area。

        WGC 給的範圍依視窗樣式而定：無邊框時就是 client area，有標題列與邊框時
        是整個視窗。兩種都要處理，否則視窗化之後 ROI 會整個偏掉。
        """
        frame = self.latest()
        if frame is None:
            return None
        return crop_to_client(frame, self.hwnd)


def crop_to_client(frame: np.ndarray, hwnd: int) -> np.ndarray:
    """把視窗畫面裁成 client area。尺寸已經相符就原樣回傳。"""
    height, width = frame.shape[:2]
    client_width, client_height = get_client_size(hwnd)
    if client_width <= 0 or client_height <= 0:
        return frame
    if (width, height) == (client_width, client_height):
        return frame

    left, top, right, bottom = window_rect(hwnd)
    origin_x, origin_y = client_origin_on_screen(hwnd)
    offset_x = max(0, origin_x - left)
    offset_y = max(0, origin_y - top)

    # 視窗剛好在縮放中時尺寸可能對不上，夾一下避免裁出空陣列。
    offset_x = min(offset_x, max(0, width - 1))
    offset_y = min(offset_y, max(0, height - 1))
    end_x = min(width, offset_x + client_width)
    end_y = min(height, offset_y + client_height)
    if end_x - offset_x < 2 or end_y - offset_y < 2:
        return frame
    return frame[offset_y:end_y, offset_x:end_x]
