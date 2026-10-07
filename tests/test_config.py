"""設定與 ROI 座標模型測試。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from mapleexp.config import (
    SUGGESTED_EXP_ROI,
    Config,
    Roi,
)


class TestRoiAnchoring(unittest.TestCase):
    def test_round_trip_at_same_size(self):
        rect = (1117, 1042, 1189, 1057)
        roi = Roi.from_pixels(rect, 1920, 1080, anchor="bottom-center")
        self.assertEqual(roi.to_pixels(1920, 1080), rect)

    def test_bottom_center_survives_window_resize(self):
        """狀態列是固定像素大小、錨定在底部中央 —— 換視窗大小後位移必須不變。

        這是選用錨點模型而不是相對比例的全部理由，所以要測得很明確。
        """
        rect = (1117, 1042, 1189, 1057)
        roi = Roi.from_pixels(rect, 1920, 1080, anchor="bottom-center")

        # 視窗縮成 1600x900：狀態列跟著底部中央移動，但大小不變。
        left, top, right, bottom = roi.to_pixels(1600, 900)
        self.assertEqual(right - left, 1189 - 1117)
        self.assertEqual(bottom - top, 1057 - 1042)
        # 距離底部中央的相對位置完全一樣
        self.assertEqual(left - 1600 // 2, 1117 - 1920 // 2)
        self.assertEqual(top - 900, 1042 - 1080)

    def test_relative_model_would_have_broken(self):
        """對照組：如果用相對比例存，同一塊區域換解析度就會跑掉。"""
        rect = (1117, 1042, 1189, 1057)
        fraction_x = rect[0] / 1920
        self.assertNotAlmostEqual(fraction_x * 1600, 1600 // 2 + (1117 - 960), places=0)

    def test_top_left_anchor_is_plain_pixels(self):
        roi = Roi(anchor="top-left", dx=10, dy=20, w=30, h=40)
        self.assertEqual(roi.to_pixels(800, 600), (10, 20, 40, 60))

    def test_clamped_to_frame(self):
        roi = Roi(anchor="top-left", dx=-50, dy=-50, w=10_000, h=10_000)
        left, top, right, bottom = roi.to_pixels(100, 80)
        self.assertEqual((left, top), (0, 0))
        self.assertEqual((right, bottom), (100, 80))

    def test_unset_roi(self):
        self.assertFalse(Roi().is_set())
        self.assertTrue(Roi(dx=0, dy=-10, w=5, h=5).is_set())

    def test_unknown_anchor_falls_back(self):
        roi = Roi(anchor="nonsense", dx=0, dy=-10, w=10, h=10)
        self.assertEqual(roi.to_pixels(200, 100), Roi(anchor="bottom-center", dx=0, dy=-10, w=10, h=10).to_pixels(200, 100))

    def test_suggested_roi_lands_on_the_status_bar(self):
        """實測值：1920x1080 時應落在狀態列文字那一列。"""
        left, top, right, bottom = SUGGESTED_EXP_ROI.to_pixels(1920, 1080)
        self.assertTrue(1100 <= left <= 1130, left)
        self.assertTrue(1180 <= right <= 1210, right)
        self.assertTrue(1038 <= top <= 1046, top)
        self.assertTrue(1052 <= bottom <= 1062, bottom)


class TestConfigPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg = Config()
            cfg.reader.exp_roi = Roi(anchor="bottom-center", dx=153, dy=-39, w=81, h=17)
            cfg.reader.threshold = 177
            cfg.reader.ink_color = [240, 240, 240]
            cfg.tracker.sample_interval = 2.5
            cfg.tracker.rate_windows = [60, 600]
            cfg.save(path)

            loaded = Config.load(path)
            self.assertEqual(loaded.reader.exp_roi, cfg.reader.exp_roi)
            self.assertEqual(loaded.reader.threshold, 177)
            self.assertEqual(loaded.reader.ink_color, [240, 240, 240])
            self.assertEqual(loaded.tracker.sample_interval, 2.5)
            self.assertEqual(loaded.tracker.rate_windows, [60, 600])
            self.assertTrue(loaded.is_calibrated())

    def test_missing_file_gives_defaults(self):
        cfg = Config.load(Path("no-such-config.json"))
        self.assertFalse(cfg.is_calibrated())

    def test_corrupt_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("}{", encoding="utf-8")
            self.assertFalse(Config.load(path).is_calibrated())

    def test_legacy_relative_roi_is_discarded(self):
        """0.1 之前的相對比例格式沿用會得到看似合法但完全錯誤的座標。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"reader": {"exp_roi": {"x": 0.58, "y": 0.96, "w": 0.04, "h": 0.015}}}),
                encoding="utf-8",
            )
            cfg = Config.load(path)
            self.assertFalse(cfg.is_calibrated())

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"reader": {"threshold": 99, "future_option": True}}),
                encoding="utf-8",
            )
            self.assertEqual(Config.load(path).reader.threshold, 99)


if __name__ == "__main__":
    unittest.main()
