"""OCR 前處理測試。

測的是把畫面整理成「黑字白底」的那一步，不需要 OCR 引擎，所以在任何機器上都跑
得起來。獨立成一個檔案是因為它擋的是這個專案裡最貴的一個錯：**用亮度門檻去挑字**。

這個遊戲的 UI 只用三種顏色（底色／白字／黑陰影），但畫面上一半以上的像素是它們的
混合（抗鋸齒）。白字和黑陰影各半混出來的灰，亮度剛好跟淺色底差不多 —— 亮度門檻
會把它判成底色，在筆畫中間挖出洞。字越小筆畫越密，挖掉的越多，最後整行讀不出來。
改成把三色混合解回來（算出文字覆蓋率）就拿得回那些資訊。
"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
import numpy as np

from mapleexp.vision.ocr import (
    BLACK,
    WHITE,
    bilinear,
    coverage,
    dominant_colour,
    lines_of,
    prepare,
    tidy,
)

MARGIN = 28

# 實機量到的三組調色盤（BGR）。用真的顏色測，不要自己編灰階 —— 灰階會讓
# 「文字−底色」跟「陰影−底色」在 RGB 空間裡共線，那是另一條程式路徑。
MINIMAP_BG = (204, 187, 153)   # 小地圖面板的淺鋼藍
STATUS_BG = (68, 63, 54)       # 狀態列底下的深色
TEXT = (255, 255, 255)
SHADOW = (0, 0, 0)


def crop(background, text, outline=None) -> np.ndarray:
    """做一塊 40x20 的假 UI：底色佔多數，中間一條文字，可選深色描邊。

    每個顏色可以給 BGR tuple，也可以給單一灰階值。
    """

    def as_bgr(value):
        return (value, value, value) if np.isscalar(value) else tuple(value)

    image = np.full((20, 40, 4), 255, dtype=np.uint8)
    image[..., :3] = as_bgr(background)
    if outline is not None:
        image[7:14, 8:32, :3] = as_bgr(outline)
    image[8:13, 10:30, :3] = as_bgr(text)
    return image


def inner(prepared: np.ndarray) -> np.ndarray:
    """去掉四周的白邊，只留放大後的內容。"""
    return prepared[MARGIN:-MARGIN, MARGIN:-MARGIN]


class TestDominantColour(unittest.TestCase):
    def test_picks_the_majority_not_the_extreme(self):
        """底色是面積最大的顏色，不是最暗或最亮的那個。"""
        found = dominant_colour(crop(MINIMAP_BG, TEXT, outline=SHADOW))
        self.assertTrue(np.allclose(found, MINIMAP_BG, atol=8), found)

    def test_dark_background(self):
        self.assertTrue(np.allclose(dominant_colour(crop(STATUS_BG, TEXT, outline=SHADOW)), STATUS_BG, atol=8))

    def test_white_background(self):
        self.assertTrue(np.allclose(dominant_colour(crop(TEXT, SHADOW)), TEXT, atol=8))

    def test_slight_gradient_does_not_split_the_vote(self):
        """底色有一點漸層時，精確眾數會碎掉，量化之後才找得到。"""
        image = crop(MINIMAP_BG, TEXT, outline=SHADOW)
        noise = np.tile(np.arange(image.shape[1]) % 3, (image.shape[0], 1))
        image[..., :3] = np.clip(image[..., :3].astype(int) + noise[..., None], 0, 255)
        self.assertTrue(np.allclose(dominant_colour(image), (205, 188, 154), atol=8))


class TestCoverage(unittest.TestCase):
    """解混合：把每個像素還原成 文字／陰影／底色 的比例。"""

    def test_pure_text_is_full_coverage(self):
        alpha = coverage(crop(MINIMAP_BG, TEXT, outline=SHADOW))
        self.assertAlmostEqual(float(alpha[10, 20]), 1.0, delta=0.05)

    def test_background_is_zero_coverage(self):
        alpha = coverage(crop(MINIMAP_BG, TEXT, outline=SHADOW))
        self.assertAlmostEqual(float(alpha[1, 1]), 0.0, delta=0.05)

    def test_shadow_is_not_mistaken_for_text(self):
        alpha = coverage(crop(MINIMAP_BG, TEXT, outline=SHADOW))
        self.assertLess(float(alpha[7, 9]), 0.2)

    def test_half_text_half_shadow_is_recovered(self):
        """這是亮度門檻會失敗的那一格：白字和黑陰影各半，亮度剛好像底色。

        混出來是 (127,127,127)，亮度 127，而底色亮度 180 —— 亮度門檻會判成
        「比底色暗」也就是沒有字；解混合知道它有一半是字。
        """
        image = crop(MINIMAP_BG, TEXT, outline=SHADOW)
        image[10, 20, :3] = 127          # 白字與黑陰影各半混出來的灰
        alpha = coverage(image)
        self.assertGreater(float(alpha[10, 20]), 0.35)

    def test_dark_text_on_white(self):
        """面板標題：黑字白底，文字與陰影對調。"""
        alpha = coverage(crop(TEXT, SHADOW), text=BLACK, shadow=WHITE)
        self.assertGreater(float(alpha[10, 20]), 0.9)
        self.assertLess(float(alpha[1, 1]), 0.1)

    def test_text_the_same_colour_as_the_background_yields_nothing(self):
        alpha = coverage(np.full((10, 10, 4), 255, dtype=np.uint8))
        self.assertEqual(float(alpha.max()), 0.0)

    def test_grey_palette_falls_back_to_a_projection(self):
        """純灰階的 UI：三個顏色共線，文字／陰影的拆分在數學上不唯一。

        這時候不該假裝解得出來，退回「比底色偏文字方向的算字」這個約定 —— 也就是
        舊的亮度做法。顏色在這個情況下本來就沒有多餘的資訊可用。
        """
        alpha = coverage(crop(128, 255, outline=0))
        self.assertGreater(float(alpha[10, 20]), 0.9)   # 純白的字
        self.assertEqual(float(alpha[1, 1]), 0.0)       # 底色
        self.assertEqual(float(alpha[7, 9]), 0.0)       # 陰影不算字

    def test_explicit_background_overrides_detection(self):
        alpha = coverage(crop(MINIMAP_BG, TEXT, outline=SHADOW), background=MINIMAP_BG)
        self.assertAlmostEqual(float(alpha[10, 20]), 1.0, delta=0.05)


class TestPrepare(unittest.TestCase):
    def _contrast(self, image, bright_text=True):
        out = inner(prepare(image, scale=2, bright_text=bright_text)).astype(float)
        # 中央是文字、四角是底色。
        text = out[out.shape[0] // 2, out.shape[1] // 2]
        corner = out[2, 2]
        return corner - text

    def test_bright_text_on_dark_background(self):
        """狀態列的角色名：一直都能用，不要改壞。"""
        self.assertGreater(self._contrast(crop(STATUS_BG, TEXT, outline=SHADOW)), 200)

    def test_bright_text_on_bright_background(self):
        """小地圖的地圖名：白字＋深色描邊壓在淺鋼藍底上。

        這是會回歸的那一個。舊做法以整張圖的最暗點（描邊）為基準再反相，字跟底
        都壓成深色，對比幾乎歸零，實機直接回傳空字串。
        """
        self.assertGreater(self._contrast(crop(MINIMAP_BG, TEXT, outline=SHADOW)), 200)

    def test_shadow_does_not_become_ink(self):
        """深色陰影是字的輪廓，不是字本身；當成墨水會把筆畫糊成一團。"""
        out = inner(prepare(crop(MINIMAP_BG, TEXT, outline=SHADOW), scale=1)).astype(float)
        self.assertGreater(out[7, 9], 200)   # 陰影位置應該還是白的

    def test_dark_text_on_white_background(self):
        """面板標題：純白底黑字，走 bright_text=False 那條路。"""
        self.assertGreater(
            self._contrast(crop(TEXT, SHADOW), bright_text=False), 200
        )

    def test_output_is_black_text_on_white(self):
        out = inner(prepare(crop(MINIMAP_BG, TEXT, outline=SHADOW), scale=1))
        self.assertEqual(out.max(), 255)  # 底色被拉到純白
        self.assertLess(int(out.min()), 40)  # 文字被壓到接近純黑

    def test_margin_is_white_and_the_right_size(self):
        out = prepare(crop(STATUS_BG, TEXT, outline=SHADOW), scale=2, margin=MARGIN)
        self.assertEqual(out.shape, (20 * 2 + MARGIN * 2, 40 * 2 + MARGIN * 2))
        self.assertTrue((out[:MARGIN, :] == 255).all())
        self.assertTrue((out[:, :MARGIN] == 255).all())

    def test_flat_image_does_not_blow_up(self):
        """整塊同色時沒有文字可言，重點是不要除以零。"""
        out = prepare(np.full((10, 10, 4), 128, dtype=np.uint8), scale=2)
        self.assertTrue(np.isfinite(out).all())

    def test_scale(self):
        out = prepare(crop(STATUS_BG, TEXT, outline=SHADOW), scale=3, margin=0)
        self.assertEqual(out.shape, (60, 120))


class FakeLine:
    def __init__(self, text):
        self.text = text


class FakeResult:
    """模仿 OcrResult：``text`` 是被攤平的一條，``lines`` 才有行結構。"""

    def __init__(self, lines):
        self.lines = [FakeLine(t) for t in lines]
        self.text = " ".join(lines)


class TestLineStructure(unittest.TestCase):
    """擋住一個很容易再犯的錯：``OcrResult.text`` 裡面**沒有換行**。

    中文辨識每個字之間都會插空白，所以把空白清掉之後，視覺上的兩行就黏成一條。
    小地圖面板「第一行地區、第二行地圖名」就是靠這個換行分開的。
    """

    def test_lines_are_joined_with_newlines(self):
        result = FakeResult(["迷 霧 森 林", "奇 幻 村"])
        self.assertEqual(lines_of(result), "迷 霧 森 林\n奇 幻 村")

    def test_result_text_alone_would_lose_the_break(self):
        """示範問題本身：直接用 .text 會攤平成一條。"""
        self.assertNotIn("\n", FakeResult(["迷 霧 森 林", "奇 幻 村"]).text)

    def test_tidy_strips_spaces_but_keeps_the_break(self):
        text = tidy(lines_of(FakeResult(["迷 霧 森 林", "奇 幻 村"])))
        self.assertEqual(text.splitlines(), ["迷霧森林", "奇幻村"])

    def test_falls_back_to_text_when_there_are_no_lines(self):
        result = FakeResult([])
        result.text = "備 援"
        self.assertEqual(lines_of(result), "備 援")

    def test_broken_result_object_is_not_an_error(self):
        class Broken:
            @property
            def lines(self):
                raise RuntimeError("boom")

            @property
            def text(self):
                raise RuntimeError("boom")

        self.assertEqual(lines_of(Broken()), "")

    def test_tidy_drops_blank_lines(self):
        self.assertEqual(tidy("甲 乙\n\n丙"), "甲乙\n丙")


class TestBilinear(unittest.TestCase):
    def test_size(self):
        self.assertEqual(bilinear(np.zeros((3, 5)), 4).shape, (12, 20))

    def test_interpolates_rather_than_duplicating(self):
        """最近鄰會把筆畫變成方塊，OCR 辨識率掉得很明顯。"""
        out = bilinear(np.array([[0.0, 100.0]]), 4)
        middle = sorted(set(np.round(out[0], 3)))
        self.assertGreater(len(middle), 2, "應該要有中間值，不是只有 0 跟 100")

    def test_scale_one_is_a_passthrough(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(bilinear(source, 1), source))


if __name__ == "__main__":
    unittest.main()
