"""自動定位經驗值欄位的測試。

用合成畫面：底部放一條狀態列，周圍塞進會干擾的東西（亮塊、另一個長得像
HP 的括號欄位），確認定位只會鎖定真正的經驗值欄位。
"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from mapleexp.vision.locate import _exp_span, locate_exp_field


def make_screen(
    text: str = "123456[12.34%]",
    width: int = 420,
    height: int = 300,
    text_x: int = 180,
    text_y: int = 270,
    clutter: bool = True,
    extra: str | None = "1446[2110]",
    extra_x: int = 40,
) -> np.ndarray:
    """合成一張「遊戲畫面」：底部是狀態列，上面有干擾物。"""
    screen = np.zeros((height, width, 4), dtype=np.uint8)
    screen[..., :3] = (30, 26, 22)
    screen[..., 3] = 255

    if clutter:
        rng = np.random.default_rng(7)
        # 上半部塞一些亮塊，模擬遊戲場景；也放幾條亮橫線模擬面板邊框。
        for _ in range(60):
            y = int(rng.integers(0, height - 40))
            x = int(rng.integers(0, width - 20))
            h = int(rng.integers(3, 18))
            w = int(rng.integers(3, 18))
            screen[y : y + h, x : x + w, :3] = int(rng.integers(90, 255))
        screen[text_y - 6, :, :3] = 240
        screen[text_y + 12, :, :3] = 215

    def stamp(value: str, left: int, top: int) -> None:
        patch = testfont.render_bgra(value, padding=0)
        mask = testfont.render_mask(value, padding=0)
        h, w = mask.shape
        region = screen[top : top + h, left : left + w, :3]
        region[mask] = patch[..., :3][mask]

    if extra is not None:
        stamp(extra, extra_x, text_y)
    stamp(text, text_x, text_y)
    return screen


class TestExpSpan(unittest.TestCase):
    def test_picks_bracketed_percentage_field(self):
        left = list("14462110")
        labels = left + list("123456[12.34%]")
        # 兩個欄位之間有空隙 -> 不可以把左邊的數字吃進來
        breaks = {len(left)}
        span = _exp_span(labels, breaks)
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual("".join(labels[start:end]), "123456[12.34%]")

    def test_without_gap_information_it_would_overreach(self):
        """沒有空隙資訊時會多吃到隔壁欄位 —— 正是要靠 breaks 擋掉的狀況。"""
        labels = list("14462110") + list("123456[12.34%]")
        start, _end = _exp_span(labels)
        self.assertEqual(start, 0)

    def test_no_percentage_means_no_match(self):
        self.assertIsNone(_exp_span(list("14462110")))

    def test_requires_digits_before_bracket(self):
        # 只有括號與百分比、前面沒有數字 -> 不是經驗值欄位
        self.assertIsNone(_exp_span(list("[12.34%]")))

    def test_handles_merged_decimal_label(self):
        labels = ["1", "2", "3", "[", "1", "2", ".3", "4", "%", "]"]
        span = _exp_span(labels)
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(start, 0)
        self.assertEqual(end, len(labels))


class TestLocate(unittest.TestCase):
    def setUp(self):
        self.templates = testfont.build_template_set()

    def test_finds_field_in_cluttered_screen(self):
        screen = make_screen()
        found = locate_exp_field(screen, self.templates)
        self.assertIsNotNone(found, "在有干擾的畫面上找不到經驗值欄位")
        self.assertEqual(found.text, "123456[12.34%]")
        self.assertEqual(found.exp_abs, 123456)
        self.assertEqual(found.exp_pct, 12.34)
        self.assertTrue(found.confident)

    def test_roi_maps_back_to_the_text(self):
        screen = make_screen()
        found = locate_exp_field(screen, self.templates)
        left, top, right, bottom = found.rect
        # ROI 應該剛好框住那串字（允許幾個像素的留邊）
        self.assertTrue(175 <= left <= 182, left)
        self.assertTrue(bottom - top <= 14, bottom - top)
        # 換算成錨點座標再換算回來要一致
        again = found.roi.to_pixels(screen.shape[1], screen.shape[0])
        self.assertEqual(again, found.rect)

    def test_roi_survives_resolution_change(self):
        """同一個 ROI 換到別的畫面大小仍然指到距離底部中央一樣的位置。"""
        found = locate_exp_field(make_screen(), self.templates)
        small = found.roi.to_pixels(300, 200)
        big = found.roi.to_pixels(420, 300)
        self.assertEqual(small[2] - small[0], big[2] - big[0])
        self.assertEqual(small[0] - 300 // 2, big[0] - 420 // 2)

    def test_ignores_hp_style_field_without_percentage(self):
        screen = make_screen(text="999[7.50%]", extra="1446[2110]")
        found = locate_exp_field(screen, self.templates)
        self.assertIsNotNone(found)
        self.assertEqual(found.exp_abs, 999)

    def test_returns_none_when_no_exp_field(self):
        screen = make_screen(text="1446[2110]", extra=None)
        self.assertIsNone(locate_exp_field(screen, self.templates))

    def test_works_without_clutter(self):
        screen = make_screen(clutter=False, extra=None)
        found = locate_exp_field(screen, self.templates)
        self.assertIsNotNone(found)
        self.assertEqual(found.exp_abs, 123456)


if __name__ == "__main__":
    unittest.main()
