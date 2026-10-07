"""自動設定：已存的 ROI 要先試讀，讀不到就重新定位。

真實案例：1920x1080 定位好的 ROI 在 1366x768 偏了 6 像素，第一個數字被切掉
一半。以前看到有 ROI 就直接沿用，於是每一格都辨識失敗。
"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from mapleexp.autosetup import auto_configure
from mapleexp.config import Config, Roi
from mapleexp.win32.capture import CaptureError, Frame
from test_locate import make_screen

# 開頭用 6：測試字型的 1 太窄，切掉一半還是認得出來，測不到「偏了就壞」。
TEXT = "623456[12.34%]"


class ScreenCapturer:
    """把一張合成畫面當成遊戲視窗。"""

    def __init__(self, screen: np.ndarray | None) -> None:
        self.screen = screen

    def capture_client(self) -> Frame:
        if self.screen is None:
            raise CaptureError("視窗不見了")
        return Frame(pixels=self.screen, backend="fake")

    def capture_roi(self, roi: Roi) -> Frame:
        frame = self.capture_client()
        h, w = frame.pixels.shape[:2]
        left, top, right, bottom = roi.to_pixels(w, h)
        return Frame(pixels=frame.pixels[top:bottom, left:right], backend="fake")


class TestAutoConfigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = testfont.build_template_set()
        cls.screen = make_screen(TEXT)

    def located_config(self) -> Config:
        """先用自動定位拿到一組正確的設定，當作「上次存下來的」。"""
        cfg = Config()
        result = auto_configure(cfg, ScreenCapturer(self.screen), self.templates)
        self.assertEqual(result.roi_source, "自動偵測", result.message)
        return cfg

    def test_saved_roi_that_still_reads_is_kept(self):
        cfg = self.located_config()
        before = cfg.reader.exp_roi
        result = auto_configure(cfg, ScreenCapturer(self.screen), self.templates)
        self.assertEqual(result.roi_source, "既有設定")
        self.assertEqual(cfg.reader.exp_roi, before)

    def test_shifted_roi_is_relocated(self):
        """ROI 偏到把第一個數字切成一半 —— 那個字認不出來，就該重新定位。

        偏移量要落在字形中間。偏得剛好整個數字都掉出去，讀到的是「少一位數」
        但完全合法的字串，這個檢查抓不到 —— 那是另一個問題，這裡不假裝解決。
        """
        cfg = self.located_config()
        good = cfg.reader.exp_roi
        # 定位到的 ROI 左緣在第一個字前 1 像素；字寬 5，往右 4 像素剛好切在中間。
        cfg.reader.exp_roi = Roi(anchor=good.anchor, dx=good.dx + 4, dy=good.dy, w=good.w, h=good.h)
        result = auto_configure(cfg, ScreenCapturer(self.screen), self.templates)
        self.assertEqual(result.roi_source, "自動偵測", result.message)
        self.assertIn("重新定位", result.message)
        self.assertEqual(cfg.reader.exp_roi, good)

    def test_unreadable_roi_is_kept_when_relocation_fails(self):
        """還沒進遊戲：讀不到、也找不到。不能拒絕啟動，要沿用舊的等它出現。"""
        cfg = self.located_config()
        saved = cfg.reader.exp_roi
        blank = np.zeros_like(self.screen)
        blank[..., 3] = 255
        result = auto_configure(cfg, ScreenCapturer(blank), self.templates)
        self.assertTrue(result.ok)
        self.assertEqual(result.roi_source, "既有設定")
        self.assertIn("目前讀不到", result.message)
        self.assertEqual(cfg.reader.exp_roi, saved)

    def test_capture_failure_with_saved_roi_is_not_fatal(self):
        cfg = self.located_config()
        result = auto_configure(cfg, ScreenCapturer(None), self.templates)
        self.assertTrue(result.ok)
        self.assertEqual(result.roi_source, "既有設定")

    def test_no_saved_roi_and_nothing_on_screen_is_fatal(self):
        cfg = Config()
        blank = np.zeros_like(self.screen)
        result = auto_configure(cfg, ScreenCapturer(blank), self.templates)
        self.assertFalse(result.ok)

    def test_force_relocates_even_when_saved_roi_reads(self):
        cfg = self.located_config()
        result = auto_configure(cfg, ScreenCapturer(self.screen), self.templates, force=True)
        self.assertEqual(result.roi_source, "自動偵測")


if __name__ == "__main__":
    unittest.main()
