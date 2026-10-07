"""視窗畫面擷取。

兩個後端，依可靠度排序：

``wgc``（預設，需要 ``windows-capture``）
    Windows Graphics Capture。要的是視窗在 DWM 裡的合成結果，所以
    **視窗模式、全螢幕、被別的視窗蓋住都抓得到**。OBS 的視窗擷取走的也是這條路。

``screen``
    從桌面 DC BitBlt。很便宜，但只在那塊區域沒被蓋住時才正確 ——
    被蓋住時抓到的是上層視窗的像素，所以每次都會先用 :func:`is_region_visible`
    驗證歸屬。沒裝 windows-capture 時用這個。

``auto``（預設）依序往下退，全部失敗才回報擷取失敗 ——
讓追蹤器進入暫停，而不是記下錯誤的數字。

**為什麼沒有 PrintWindow。** 曾經有第三個後端 ``printwindow``，請視窗自己畫到我們
的 DC。它對這個遊戲從來沒成功過（Unity + D3D11 直接回傳失敗），而且它是這整個程式
唯一會**對遊戲視窗送訊息**（WM_PRINT）的地方 —— 上面兩個後端一個透過 DWM、一個讀
桌面 DC，都不會碰到遊戲程序。一條沒用、又是唯一會碰到遊戲的路，拿掉。
舊設定檔若還寫著 ``printwindow`` 會自動當成 ``auto``。
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from ..config import Roi
from .api import (
    BI_RGB,
    BITMAPINFO,
    DIB_RGB_COLORS,
    SRCCOPY,
    enable_dpi_awareness,
    gdi32,
    user32,
)
from .wgc import WGC_AVAILABLE, WgcError, WgcSession
from .windows import (
    client_origin_on_screen,
    get_client_size,
    is_minimized,
    is_region_visible,
    window_exists,
)

# 取樣後的像素極差小於此值就視為「空白畫面」（後端沒真的畫出東西）。
BLANK_RANGE_THRESHOLD = 4

BACKENDS = ("auto", "wgc", "screen")


class CaptureError(RuntimeError):
    """擷取失敗。訊息會直接顯示給使用者，所以要寫得具體。"""


@dataclass
class Frame:
    """一張 BGRA 影像。"""

    pixels: np.ndarray  # shape (h, w, 4), dtype uint8, 順序為 B G R A
    backend: str

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


class _DibBuffer:
    """可重複使用的記憶體 DC + DIB section。"""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        screen_dc = user32.GetDC(None)
        if not screen_dc:
            raise CaptureError("無法取得桌面 DC")
        try:
            self.dc = gdi32.CreateCompatibleDC(screen_dc)
        finally:
            user32.ReleaseDC(None, screen_dc)
        if not self.dc:
            raise CaptureError("無法建立記憶體 DC")

        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(info.bmiHeader)
        info.bmiHeader.biWidth = width
        # 負高度 = top-down，省掉之後上下翻轉。
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB

        self.bits = ctypes.c_void_p()
        self.bitmap = gdi32.CreateDIBSection(
            self.dc,
            ctypes.byref(info),
            DIB_RGB_COLORS,
            ctypes.byref(self.bits),
            None,
            0,
        )
        if not self.bitmap or not self.bits:
            gdi32.DeleteDC(self.dc)
            raise CaptureError(f"無法建立 {width}x{height} 的 DIB section")
        self.previous = gdi32.SelectObject(self.dc, self.bitmap)
        self.size = width * height * 4

    def to_array(self) -> np.ndarray:
        raw = ctypes.string_at(self.bits, self.size)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)

    def close(self) -> None:
        if getattr(self, "dc", None):
            if getattr(self, "previous", None):
                gdi32.SelectObject(self.dc, self.previous)
            if getattr(self, "bitmap", None):
                gdi32.DeleteObject(self.bitmap)
            gdi32.DeleteDC(self.dc)
            self.dc = None
            self.bitmap = None


class WindowCapturer:
    """針對單一視窗的擷取器。DC、緩衝區與 WGC 工作階段都會重複使用。"""

    def __init__(self, hwnd: int, backend: str = "auto") -> None:
        enable_dpi_awareness()
        self.hwnd = int(hwnd)
        self.backend = backend if backend in BACKENDS else "auto"
        self.last_backend = ""
        self._buffers: dict[tuple[int, int], _DibBuffer] = {}
        self._wgc: WgcSession | None = None
        self._wgc_failed = ""

    # ------------------------------------------------------------------ #
    # 公開介面
    # ------------------------------------------------------------------ #

    def client_size(self) -> tuple[int, int]:
        return get_client_size(self.hwnd)

    def capture_client(self) -> Frame:
        """抓整個 client area。校準時用。"""
        return self._capture(None)

    def capture_roi(self, roi: Roi) -> Frame:
        """只抓 ROI。追蹤時用這個。"""
        if not roi.is_set():
            raise CaptureError("ROI 尚未校準")
        return self._capture(roi)

    def close(self) -> None:
        if self._wgc is not None:
            self._wgc.stop()
            self._wgc = None
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()

    def __enter__(self) -> "WindowCapturer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 內部
    # ------------------------------------------------------------------ #

    def _require_client_size(self) -> tuple[int, int]:
        if not window_exists(self.hwnd):
            raise CaptureError("遊戲視窗已關閉")
        if is_minimized(self.hwnd):
            raise CaptureError("遊戲視窗已最小化，無法擷取")
        width, height = get_client_size(self.hwnd)
        if width <= 0 or height <= 0:
            raise CaptureError("遊戲視窗大小為零")
        return width, height

    def _capture(self, roi: Roi | None) -> Frame:
        width, height = self._require_client_size()
        problems: list[str] = []

        for backend in self._backend_order():
            try:
                pixels = self._capture_with(backend, roi, width, height, problems)
            except CaptureError as exc:
                problems.append(f"{backend}: {exc}")
                continue
            if pixels is None:
                continue
            if _looks_blank(pixels):
                problems.append(f"{backend}: 擷取結果為空白畫面")
                continue
            self.last_backend = backend
            return Frame(pixels=pixels, backend=backend)

        raise CaptureError("；".join(problems) or "擷取失敗")

    def _capture_with(
        self,
        backend: str,
        roi: Roi | None,
        width: int,
        height: int,
        problems: list[str],
    ) -> np.ndarray | None:
        if backend == "wgc":
            frame = self._wgc_client_frame()
            if frame is None:
                problems.append(f"wgc: {self._wgc_failed or '尚未取得畫面'}")
                return None
            if roi is None:
                return frame
            # 用這一幀自己的尺寸換算 ROI，而不是 GetClientRect ——
            # 視窗正在縮放時兩者可能差一兩個像素。
            left, top, right, bottom = roi.to_pixels(frame.shape[1], frame.shape[0])
            return frame[top:bottom, left:right]

        rect = (0, 0, width, height) if roi is None else roi.to_pixels(width, height)
        left, top, right, bottom = rect
        origin = client_origin_on_screen(self.hwnd)
        screen_rect = (
            origin[0] + left,
            origin[1] + top,
            origin[0] + right,
            origin[1] + bottom,
        )

        if backend == "screen":
            if not is_region_visible(self.hwnd, screen_rect):
                problems.append("screen: 擷取區域被其他視窗遮住")
                return None
            return self._blit_screen(screen_rect)

        raise CaptureError(f"未知的擷取後端：{backend}")

    def _backend_order(self) -> list[str]:
        if self.backend != "auto":
            return [self.backend]
        order = []
        if WGC_AVAILABLE and not self._wgc_failed:
            order.append("wgc")
        order.append("screen")
        return order

    # ---------------------------- WGC ---------------------------------- #

    def _wgc_client_frame(self) -> np.ndarray | None:
        if not WGC_AVAILABLE:
            self._wgc_failed = "未安裝 windows-capture"
            return None

        if self._wgc is None:
            try:
                self._wgc = WgcSession(self.hwnd)
                self._wgc.start()
            except WgcError as exc:
                self._wgc = None
                self._wgc_failed = str(exc)
                return None
            # 第一幀要等 DWM 把合成結果送過來。
            if not self._wgc.wait_for_first_frame(3.0):
                self._wgc.stop()
                self._wgc = None
                self._wgc_failed = "3 秒內沒有收到畫面"
                return None

        if self._wgc.closed:
            self._wgc.stop()
            self._wgc = None
            self._wgc_failed = "擷取工作階段已關閉（視窗可能已關閉）"
            return None

        return self._wgc.client_frame()

    # ---------------------------- GDI ---------------------------------- #

    def _buffer(self, width: int, height: int) -> _DibBuffer:
        key = (width, height)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = _DibBuffer(width, height)
            self._buffers[key] = buffer
            # 視窗反覆縮放可能累積很多尺寸，保留最近幾個就好。
            if len(self._buffers) > 4:
                oldest = next(iter(self._buffers))
                if oldest != key:
                    self._buffers.pop(oldest).close()
        return buffer

    def _blit_screen(self, screen_rect: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = screen_rect
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            raise CaptureError("擷取區域大小無效")
        buffer = self._buffer(width, height)
        screen_dc = user32.GetDC(None)
        if not screen_dc:
            raise CaptureError("無法取得桌面 DC")
        try:
            # 刻意不加 CAPTUREBLT：那會把 layered 視窗（包含我們自己的浮窗）
            # 一起抓進來，可能汙染 ROI。
            ok = gdi32.BitBlt(
                buffer.dc, 0, 0, width, height, screen_dc, left, top, SRCCOPY
            )
        finally:
            user32.ReleaseDC(None, screen_dc)
        if not ok:
            raise CaptureError(f"BitBlt 失敗 (GetLastError={ctypes.get_last_error()})")
        return buffer.to_array()


def _looks_blank(pixels: np.ndarray) -> bool:
    """畫面是否幾乎完全單色（代表後端沒有真的畫出內容）。"""
    if pixels.size == 0:
        return True
    # 取樣就夠了，不用掃全圖。
    sample = pixels[::4, ::4, :3]
    if sample.size == 0:
        sample = pixels[..., :3]
    return int(sample.max()) - int(sample.min()) < BLANK_RANGE_THRESHOLD
