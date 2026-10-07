"""視窗查找與幾何計算。"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

from .api import (
    GA_ROOT,
    POINT,
    PROCESS_QUERY_LIMITED_INFORMATION,
    RECT,
    SW_SHOWMINIMIZED,
    WINDOWPLACEMENT,
    WNDENUMPROC,
    kernel32,
    user32,
)


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process: str
    client_width: int
    client_height: int
    minimized: bool

    @property
    def label(self) -> str:
        return f"{self.title or '(無標題)'} [{self.process}] {self.client_width}x{self.client_height}"


def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_process_path(hwnd: int) -> str:
    """擁有這個視窗的行程的完整執行檔路徑；取不到時回傳空字串。

    會需要完整路徑是因為要從遊戲安裝位置讀客戶端的資料檔，而玩家裝在哪不一定
    （Steam、官方啟動器、自己搬過的目錄都有可能），問正在跑的行程最準。
    """
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value
    finally:
        kernel32.CloseHandle(handle)


def get_process_name(hwnd: int) -> str:
    return os.path.basename(get_process_path(hwnd))


def get_client_size(hwnd: int) -> tuple[int, int]:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return 0, 0
    return rect.width, rect.height


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """整個視窗（含邊框與標題列）在螢幕上的矩形。"""
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return 0, 0, 0, 0
    return rect.left, rect.top, rect.right, rect.bottom


def client_origin_on_screen(hwnd: int) -> tuple[int, int]:
    """client area 左上角在螢幕上的座標。"""
    point = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return 0, 0
    return point.x, point.y


def is_minimized(hwnd: int) -> bool:
    placement = WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(WINDOWPLACEMENT)
    if not user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        return False
    return placement.showCmd == SW_SHOWMINIMIZED


def window_exists(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def describe(hwnd: int) -> WindowInfo:
    width, height = get_client_size(hwnd)
    return WindowInfo(
        hwnd=hwnd,
        title=get_window_title(hwnd),
        process=get_process_name(hwnd),
        client_width=width,
        client_height=height,
        minimized=is_minimized(hwnd),
    )


def list_windows(min_size: int = 200) -> list[WindowInfo]:
    """列出所有可見、且 client area 夠大的頂層視窗。"""
    found: list[int] = []

    @WNDENUMPROC
    def _callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(_callback, 0)

    result: list[WindowInfo] = []
    for hwnd in found:
        info = describe(hwnd)
        if info.client_width < min_size or info.client_height < min_size:
            continue
        if not info.title:
            continue
        result.append(info)
    result.sort(key=lambda w: (-w.client_width * w.client_height, w.title))
    return result


def find_window(title_contains: str = "", process_name: str = "") -> WindowInfo | None:
    """依標題子字串或行程名稱找視窗。

    標題優先；標題找不到時才用行程名稱（遊戲改版可能改標題，但 exe 名稱較穩定）。
    """
    candidates = list_windows()

    needle = (title_contains or "").strip().casefold()
    if needle:
        for info in candidates:
            if needle in info.title.casefold():
                return info

    proc = (process_name or "").strip().casefold()
    if proc:
        for info in candidates:
            if info.process.casefold() == proc:
                return info

    return None


def covers_whole_screen(hwnd: int) -> bool:
    """視窗是不是剛好鋪滿整個主螢幕（全螢幕）。

    用途：**獨占全螢幕時，任何置頂視窗都不會被畫到畫面上** —— GPU 直接掃描輸出
    遊戲的緩衝區，桌面合成整個被跳過。擷取不受影響（WGC 要的是合成結果），
    但浮窗會看不到。與其讓人以為程式壞了，不如主動講出來。
    """
    SM_CXSCREEN, SM_CYSCREEN = 0, 1
    screen_w = user32.GetSystemMetrics(SM_CXSCREEN)
    screen_h = user32.GetSystemMetrics(SM_CYSCREEN)
    left, top, right, bottom = window_rect(hwnd)
    return (
        left <= 0 and top <= 0
        and right >= screen_w and bottom >= screen_h
        and screen_w > 0 and screen_h > 0
    )


def is_region_visible(hwnd: int, screen_rect: tuple[int, int, int, int]) -> bool:
    """檢查螢幕上這塊區域是否真的屬於這個視窗（沒有被別的視窗蓋住）。

    這件事很重要：直接從螢幕 BitBlt 很便宜也很精確，但如果有別的視窗蓋在
    ROI 上面，抓到的會是別的東西 —— 而且可能剛好是看起來合理的數字。
    先用 WindowFromPoint 驗證歸屬，就能把這個風險排除。
    """
    left, top, right, bottom = screen_rect
    if right <= left or bottom <= top:
        return False

    # 取四角稍微內縮的點，加上中心點。
    inset = 1
    probes = [
        (left + inset, top + inset),
        (right - inset - 1, top + inset),
        (left + inset, bottom - inset - 1),
        (right - inset - 1, bottom - inset - 1),
        ((left + right) // 2, (top + bottom) // 2),
    ]
    for x, y in probes:
        hit = user32.WindowFromPoint(POINT(x, y))
        if not hit:
            return False
        root = user32.GetAncestor(hit, GA_ROOT)
        if int(root or hit) != int(hwnd):
            return False
    return True
