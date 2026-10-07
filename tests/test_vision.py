"""視覺管線測試：二值化 -> 切割 -> 模板比對 -> 解析。

用內建點陣字型合成畫面，所以完全不需要遊戲在跑。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from mapleexp.core.reading import parse_exp_text
from mapleexp.vision import pngio
from mapleexp.vision.preprocess import (
    binarize,
    ink_bounds,
    otsu_threshold,
    suggest_binarization,
    to_luma,
    trim,
)
from mapleexp.vision.segment import column_runs, normalize_glyph, segment
from mapleexp.vision.templates import TemplateSet, iou, preview_lines


def recognize(
    text: str,
    *,
    gap: int = 1,
    scale: int = 1,
    noise: int = 0,
    templates: TemplateSet | None = None,
    threshold: int = 128,
) -> str:
    """跑完整條管線並回傳辨識出來的字串。"""
    templates = templates or testfont.build_template_set()
    image = testfont.render_bgra(text, gap=gap, scale=scale, noise=noise)
    mask = binarize(image, threshold=threshold, invert=False)
    boxes = segment(mask, expected_width=templates.median_width())
    out = []
    for box in boxes:
        match = templates.match(box.extract(mask))
        out.append(match.char or "?")
    return "".join(out)


class TestPreprocess(unittest.TestCase):
    def test_luma_of_known_colors(self):
        image = np.zeros((1, 3, 4), dtype=np.uint8)
        image[0, 0, :3] = (0, 0, 0)        # 黑
        image[0, 1, :3] = (255, 255, 255)  # 白
        image[0, 2, :3] = (0, 255, 0)      # 綠 (BGR)
        luma = to_luma(image)
        self.assertEqual(int(luma[0, 0]), 0)
        self.assertGreater(int(luma[0, 1]), 250)
        self.assertTrue(80 < int(luma[0, 2]) < 170)

    def test_binarize_bright_text_on_dark_background(self):
        image = testfont.render_bgra("123")
        mask = binarize(image, threshold=128, invert=False)
        self.assertTrue(mask.any())
        # 文字一定是少數像素
        self.assertLess(mask.mean(), 0.5)

    def test_binarize_dark_text_on_bright_background(self):
        image = testfont.render_bgra(
            "123", ink=(10, 10, 10), background=(240, 240, 240)
        )
        mask = binarize(image, threshold=128, invert=True)
        self.assertTrue(mask.any())
        self.assertLess(mask.mean(), 0.5)

    def test_color_key_separates_equal_luma_colors(self):
        """文字與背景亮度幾乎相同時，只有色鍵分得開。

        這就是會在狀態列上發生的狀況：背景有色塊或漸層，亮度跟文字撞在一起。
        """
        ink = (0, 130, 0)          # 暗綠
        background = (255, 0, 157)  # 紫
        image = testfont.render_bgra("123", ink=ink, background=background)
        expected = testfont.render_mask("123")

        # 前提：兩種顏色的灰階值完全相同，所以任何亮度門檻都分不開
        probe = np.zeros((1, 2, 4), dtype=np.uint8)
        probe[0, 0, :3] = ink
        probe[0, 1, :3] = background
        luma = to_luma(probe)
        self.assertEqual(int(luma[0, 0]), int(luma[0, 1]))

        # 色鍵可以精準取出文字
        color_mask = binarize(image, ink_color=list(ink), color_tolerance=60)
        self.assertTrue(np.array_equal(color_mask, expected))

        # 而亮度門檻無論怎麼調、哪個極性，都還原不出文字
        for threshold in range(256):
            for invert in (False, True):
                mask = binarize(image, threshold=threshold, invert=invert)
                self.assertFalse(np.array_equal(mask, expected))

    def test_suggest_binarization_detects_polarity(self):
        bright = testfont.render_bgra("123", ink=(255, 255, 255), background=(10, 10, 10))
        _, invert = suggest_binarization(bright)
        self.assertFalse(invert)

        dark = testfont.render_bgra("123", ink=(10, 10, 10), background=(245, 245, 245))
        _, invert = suggest_binarization(dark)
        self.assertTrue(invert)

    def test_otsu_separates_bimodal_histogram(self):
        luma = np.concatenate(
            [np.full(500, 20, dtype=np.uint8), np.full(500, 220, dtype=np.uint8)]
        )
        threshold = otsu_threshold(luma)
        self.assertTrue(20 < threshold < 220)

    def test_trim_and_bounds(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:6, 4:8] = True
        self.assertEqual(ink_bounds(mask), (4, 3, 8, 6))
        self.assertEqual(trim(mask).shape, (3, 4))

    def test_bounds_of_empty_mask(self):
        self.assertIsNone(ink_bounds(np.zeros((5, 5), dtype=bool)))


class TestSegment(unittest.TestCase):
    def test_column_runs(self):
        mask = np.zeros((5, 10), dtype=bool)
        mask[:, 1:3] = True
        mask[:, 6:9] = True
        self.assertEqual(column_runs(mask), [(1, 3), (6, 9)])

    def test_segments_each_glyph(self):
        mask = testfont.render_mask("12345")
        boxes = segment(mask)
        self.assertEqual(len(boxes), 5)
        # 字元框是 ink 的外接矩形，所以 "1" 比其他數字窄 —— 這是對的，
        # 模板比對要的就是裁切後的字形。
        self.assertLess(boxes[0].width, boxes[1].width)
        self.assertEqual([b.width for b in boxes[1:]], [testfont.GLYPH_WIDTH] * 4)
        self.assertEqual([b.left for b in boxes], sorted(b.left for b in boxes))

    def test_period_is_kept_despite_small_size(self):
        mask = testfont.render_mask("3.14")
        boxes = segment(mask)
        self.assertEqual(len(boxes), 4)

    def test_touching_glyphs_are_split_using_template_width(self):
        """字元完全黏在一起時，靠模板寬度切開。"""
        templates = testfont.build_template_set()
        mask = testfont.render_mask("12345", gap=0)
        # 確認前提：區塊數少於字元數，代表真的有字元黏在一起了。
        self.assertLess(len(column_runs(mask)), 5)
        boxes = segment(mask, expected_width=templates.median_width())
        self.assertEqual(len(boxes), 5)

    def test_normalize_glyph_trims(self):
        glyph = np.zeros((9, 9), dtype=bool)
        glyph[4:6, 2:5] = True
        self.assertEqual(normalize_glyph(glyph).shape, (2, 3))

    def test_empty_mask_yields_no_boxes(self):
        self.assertEqual(segment(np.zeros((10, 20), dtype=bool)), [])


class TestTemplateMatching(unittest.TestCase):
    def test_iou_identical_is_one(self):
        glyph = testfont.glyph_mask("8")
        self.assertAlmostEqual(iou(glyph, glyph), 1.0)

    def test_iou_tolerates_one_pixel_shift(self):
        glyph = testfont.glyph_mask("5")
        shifted = np.zeros((glyph.shape[0] + 1, glyph.shape[1] + 1), dtype=bool)
        shifted[1:, 1:] = glyph
        self.assertAlmostEqual(iou(glyph, shifted), 1.0)

    def test_every_glyph_matches_itself_best(self):
        """所有字元彼此必須可區分 —— 否則整個辨識策略就不成立。"""
        templates = testfont.build_template_set()
        for char in testfont.GLYPHS:
            match = templates.match(testfont.glyph_mask(char))
            self.assertEqual(match.char, char, f"{char!r} 被誤判為 {match.char!r}")
            self.assertAlmostEqual(match.score, 1.0)

    def test_unknown_shape_is_rejected_not_guessed(self):
        templates = testfont.build_template_set()
        blob = np.ones((7, 5), dtype=bool)
        match = templates.match(blob)
        self.assertEqual(match.char, "")

    def test_low_margin_is_rejected(self):
        """兩個模板幾乎一樣時，寧可拒絕也不要猜。"""
        templates = TemplateSet()
        a = np.zeros((7, 5), dtype=bool)
        a[1:6, 1:4] = True
        b = a.copy()
        b[3, 2] = False
        templates.add("A", a)
        templates.add("B", b)
        match = templates.match(a, min_score=0.5, min_margin=0.30)
        self.assertEqual(match.char, "")

    def test_duplicate_templates_are_not_stored_twice(self):
        templates = TemplateSet()
        self.assertTrue(templates.add("7", testfont.glyph_mask("7")))
        self.assertFalse(templates.add("7", testfont.glyph_mask("7")))
        self.assertEqual(templates.count_for("7"), 1)

    def test_multi_character_template(self):
        """黏在一起切不開的字形，直接標成多個字。

        實測到的例子：``[28.37%]`` 的小數點靠對角線碰到 3 的左下角，八連通視為
        同一塊。與其冒險去切它，不如把整塊標成 ".3"。
        """
        templates = testfont.build_template_set()
        glued = np.hstack(
            [testfont.glyph_mask("."), testfont.glyph_mask("3")]
        )
        templates.add(".3", glued)
        match = templates.match(glued)
        self.assertEqual(match.char, ".3")
        # 單一字元的比對不能被多字元模板帶壞
        self.assertEqual(templates.match(testfont.glyph_mask("3")).char, "3")

    def test_median_width_ignores_multi_character_templates(self):
        templates = testfont.build_template_set()
        templates.add(".3", np.hstack([testfont.glyph_mask("."), testfont.glyph_mask("3")]))
        self.assertEqual(templates.median_width(), float(testfont.GLYPH_WIDTH))

    def test_median_width(self):
        templates = testfont.build_template_set()
        self.assertEqual(templates.median_width(), float(testfont.GLYPH_WIDTH))

    def test_preview_lines(self):
        lines = preview_lines(testfont.glyph_mask("1"))
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[0], "..#..")


class TestTemplatePersistence(unittest.TestCase):
    def test_round_trip_preserves_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.json"
            original = testfont.build_template_set()
            original.save(path)
            loaded = TemplateSet.load(path)
            self.assertEqual(loaded.chars(), original.chars())
            self.assertEqual(len(loaded), len(original))
            for char in testfont.GLYPHS:
                self.assertEqual(loaded.match(testfont.glyph_mask(char)).char, char)
            self.assertEqual(loaded.meta.get("source"), "builtin-testfont")

    def test_missing_file_gives_empty_set(self):
        self.assertEqual(len(TemplateSet.load(Path("does-not-exist.json"))), 0)

    def test_corrupt_file_gives_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.json"
            path.write_text("{ broken", encoding="utf-8")
            self.assertEqual(len(TemplateSet.load(path)), 0)


class TestEndToEndPipeline(unittest.TestCase):
    def test_reads_classic_status_bar_layout(self):
        text = recognize("1234567[30.25%]")
        self.assertEqual(text, "1234567[30.25%]")
        self.assertEqual(parse_exp_text(text), (1234567, 30.25))

    def test_survives_scaling(self):
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                templates = TemplateSet()
                for char in testfont.GLYPHS:
                    glyph = testfont.glyph_mask(char)
                    if scale > 1:
                        glyph = np.repeat(np.repeat(glyph, scale, 0), scale, 1)
                    templates.add(char, glyph)
                out = recognize("98765[4.32%]", scale=scale, templates=templates)
                self.assertEqual(out, "98765[4.32%]")

    def test_survives_single_pixel_noise(self):
        text = recognize("55500[12.34%]", noise=6)
        self.assertEqual(parse_exp_text(text), (55500, 12.34))

    def test_touching_digits_still_parse(self):
        text = recognize("1234567[30.25%]", gap=0)
        self.assertEqual(parse_exp_text(text), (1234567, 30.25))

    def test_all_digit_strings_round_trip(self):
        for value in ("0", "10", "1000000", "90909090", "123456789012"):
            with self.subTest(value=value):
                self.assertEqual(recognize(value), value)


class TestPngIo(unittest.TestCase):
    def test_encode_is_valid_png_and_tk_readable(self):
        rgb = pngio.mask_to_rgb(testfont.render_mask("123"))
        data = pngio.encode_png(rgb)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\x0a"))
        self.assertIn(b"IHDR", data[:32])
        self.assertTrue(data.endswith(b"IEND\xae\x42\x60\x82"))

    def test_bgra_to_rgb_swaps_channels(self):
        image = np.zeros((1, 1, 4), dtype=np.uint8)
        image[0, 0] = (10, 20, 30, 255)  # B G R A
        rgb = pngio.bgra_to_rgb(image)
        self.assertEqual(tuple(rgb[0, 0]), (30, 20, 10))

    def test_scale_nearest(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertEqual(pngio.scale_nearest(rgb, 4).shape, (8, 12, 3))

    def test_to_tk_data_is_base64(self):
        rgb = pngio.mask_to_rgb(testfont.render_mask("7"))
        data = pngio.to_tk_data(rgb)
        self.assertIsInstance(data, str)
        self.assertNotIn("\n", data)


if __name__ == "__main__":
    unittest.main()
