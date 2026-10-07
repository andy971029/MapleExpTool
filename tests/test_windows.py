"""找遊戲視窗：行程名稱要優先於標題。"""

from __future__ import annotations

import unittest
from unittest import mock

import helpers  # noqa: F401

from mapleexp.win32 import windows as windows_module
from mapleexp.win32.windows import WindowInfo, find_window


def info(hwnd: int, title: str, process: str) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd, title=title, process=process,
        client_width=1366, client_height=768, minimized=False,
    )


class TestFindWindow(unittest.TestCase):
    # 真實案例：Chrome 開著這個專案的 GitHub 頁面，分頁標題含「新楓之谷：經典版」，
    # 而且在 Z 順序上排在遊戲前面。
    CANDIDATES = [
        info(1, "andy971029/MapleExpTool: 這是一個新楓之谷：經典版的經驗值追蹤工具 - Google Chrome", "chrome.exe"),
        info(2, "新楓之谷：經典版", "Maplestory_Classic.exe"),
        info(3, "MS EXP TOOL", "MapleExpTool.exe"),
    ]

    def find(self, title="新楓之谷", process="Maplestory_Classic.exe", candidates=None):
        with mock.patch.object(
            windows_module, "list_windows", return_value=candidates or self.CANDIDATES
        ):
            return find_window(title, process)

    def test_process_name_beats_browser_tab_with_matching_title(self):
        self.assertEqual(self.find().hwnd, 2)

    def test_process_match_is_case_insensitive(self):
        self.assertEqual(self.find(process="maplestory_classic.EXE").hwnd, 2)

    def test_title_is_the_fallback_when_process_not_found(self):
        """遊戲改版換了 exe 名稱時，還是要靠標題找得到。"""
        candidates = [info(1, "MS EXP TOOL", "MapleExpTool.exe"), info(2, "新楓之谷：經典版", "NewName.exe")]
        self.assertEqual(self.find(candidates=candidates).hwnd, 2)

    def test_title_only(self):
        self.assertEqual(self.find(process="").hwnd, 1)

    def test_nothing_matches(self):
        self.assertIsNone(self.find(title="不存在", process="nope.exe"))

    def test_empty_criteria(self):
        self.assertIsNone(self.find(title="", process=""))


if __name__ == "__main__":
    unittest.main()
