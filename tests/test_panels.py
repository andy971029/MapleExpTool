"""通用面板偵測的測試。

面板位置不固定（玩家可以把每個視窗拖到任何地方），所以測試刻意把同一組面板
擺在不同位置，確認偵測結果只跟內容有關、跟位置無關。
"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from mapleexp.vision.panels import (
    KNOWN_TITLES,
    find_title_bars,
    match_title,
    similarity,
    white_mask,
)


def make_frame(
    bars: list[tuple[int, int, int, int]],
    width: int = 600,
    height: int = 400,
    text: bool = True,
) -> np.ndarray:
    """合成一張畫面：雜亂的遊戲背景 + 幾條白底黑字的標題列。"""
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    rng = np.random.default_rng(5)
    frame[..., :3] = rng.integers(20, 160, size=(height, width, 3), dtype=np.uint8)
    frame[..., 3] = 255
    for left, top, right, bottom in bars:
        frame[top:bottom, left:right, :3] = 255
        if text:
            glyphs = testfont.render_mask("123", padding=1)
            h, w = glyphs.shape
            ty = top + max(0, (bottom - top - h) // 2)
            tx = left + 3
            if ty + h <= bottom and tx + w <= right:
                region = frame[ty : ty + h, tx : tx + w, :3]
                region[glyphs] = 0
    return frame


class TestWhiteMask(unittest.TestCase):
    def test_only_near_white_and_neutral(self):
        frame = np.zeros((1, 4, 4), dtype=np.uint8)
        frame[0, 0, :3] = (255, 255, 255)   # 純白
        frame[0, 1, :3] = (240, 238, 236)   # 接近白、無彩
        frame[0, 2, :3] = (255, 200, 120)   # 亮但有色 -> 不算
        frame[0, 3, :3] = (120, 120, 120)   # 灰 -> 不算
        self.assertEqual(list(white_mask(frame)[0]), [True, True, False, False])


class TestFindTitleBars(unittest.TestCase):
    def test_finds_bars_anywhere_on_screen(self):
        rects = [(40, 30, 240, 46), (300, 150, 470, 164), (120, 300, 200, 314)]
        bars = find_title_bars(make_frame(rects))
        found = {rect for rect, _fill in bars}
        for rect in rects:
            self.assertIn(rect, found, f"沒找到 {rect}")

    def test_sorted_top_down(self):
        rects = [(300, 200, 470, 214), (40, 30, 240, 46), (120, 110, 200, 124)]
        bars = find_title_bars(make_frame(rects))
        tops = [rect[1] for rect, _ in bars]
        self.assertEqual(tops, sorted(tops))

    def test_rejects_shapes_that_are_not_bars(self):
        frame = make_frame([(40, 30, 240, 46)])
        frame[100:260, 100:300, :3] = 255        # 一大塊白（太高）
        frame[320:330, 40:60, :3] = 255          # 太窄
        bars = [rect for rect, _ in find_title_bars(frame)]
        self.assertIn((40, 30, 240, 46), bars)
        self.assertEqual(len(bars), 1)

    def test_low_fill_bar_is_still_found(self):
        """「小地圖」只有三個字，黑字就吃掉快一半 —— 填充率門檻不能訂太高。

        實機量到的填充率是 0.51，而其他面板是 0.76~0.80。
        """
        frame = make_frame([(40, 30, 120, 44)])
        # 把一半塗黑，模擬字很擠的標題
        frame[32:42, 44:84, :3] = 0
        bars = find_title_bars(frame)
        self.assertTrue(any(rect[0] == 40 for rect, _ in bars))


class TestTitleMatching(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(match_title("角色資料")[0], "角色資料")
        self.assertEqual(match_title("ITEMINVENTORY")[0], "ITEM INVENTORY")

    def test_tolerates_ocr_slips(self):
        """實機 OCR 會差一兩個字；模糊比對要吸收掉。"""
        self.assertEqual(match_title("技能攔")[0], "技能欄")   # 欄 -> 攔
        self.assertEqual(match_title("小地")[0], "小地圖")      # 少一個字

    def test_rejects_unrelated_text(self):
        """背包裡的楓幣欄位也是白底黑字的細長條，必須配不上任何標題。"""
        for text in ("128,231", "", "0", "9999999"):
            self.assertEqual(match_title(text)[0], "", text)

    def test_similarity_is_symmetric_and_bounded(self):
        self.assertEqual(similarity("小地圖", "小地圖"), 1.0)
        self.assertEqual(similarity("", "小地圖"), 0.0)
        self.assertAlmostEqual(
            similarity("小地", "小地圖"), similarity("小地圖", "小地")
        )
        self.assertTrue(0.0 <= similarity("技能攔", "角色資料") < 0.5)

    def test_ambiguous_match_is_rejected(self):
        """跟兩個標題一樣像的時候寧可不認。"""
        self.assertEqual(match_title("AB", ("ABC", "ABD"))[0], "")

    def test_known_titles_are_distinct_enough(self):
        for index, left in enumerate(KNOWN_TITLES):
            for right in KNOWN_TITLES[index + 1 :]:
                self.assertLess(
                    similarity(left, right), 0.7, f"{left} 與 {right} 太像"
                )


if __name__ == "__main__":
    unittest.main()
