"""擷取後端的選擇邏輯。真正的擷取需要視窗，這裡只測不碰 Win32 的部分。"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401

from mapleexp.win32 import api, capture


class TestBackends(unittest.TestCase):
    def test_printwindow_is_gone(self):
        """PrintWindow 是唯一會對遊戲視窗送訊息（WM_PRINT）的後端，而且對這個遊戲
        從來沒成功過，已經拿掉。這個測試擋住它被不小心加回來。"""
        self.assertNotIn("printwindow", capture.BACKENDS)
        self.assertFalse(hasattr(capture.WindowCapturer, "_print_window"))
        self.assertFalse(hasattr(api, "PW_CLIENTONLY"))

    def test_legacy_printwindow_setting_falls_back_to_auto(self):
        """舊設定檔可能還寫著 printwindow，不能因此整個擷取器開不起來。"""
        capturer = capture.WindowCapturer(0, "printwindow")
        self.assertEqual(capturer.backend, "auto")

    def test_auto_order_never_touches_the_game_window(self):
        capturer = capture.WindowCapturer(0, "auto")
        order = capturer._backend_order()
        self.assertEqual(order[-1], "screen")
        self.assertTrue(set(order) <= {"wgc", "screen"})


if __name__ == "__main__":
    unittest.main()
