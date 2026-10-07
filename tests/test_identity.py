"""等級／地圖辨識的測試。"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from mapleexp.vision import identity as identity_module
from mapleexp.vision.identity import (
    LevelReader,
    LevelZone,
    MapWatcher,
    fingerprint,
    find_orange_mask,
    locate_map_zone,
    read_level_by_ocr,
)


def level_zone(text: str, image: np.ndarray | None = None) -> LevelZone:
    """用測試字型做一個等級字形區。"""
    digits = [testfont.glyph_mask(c) for c in text]
    return LevelZone(rect=(0, 0, 10, 10), digits=digits, image=image)


def orange_box() -> np.ndarray:
    """一塊假的等級方塊。內容不重要 —— 測試都把 OCR 換成假的。"""
    box = np.zeros((13, 22, 4), dtype=np.uint8)
    box[..., :3] = (17, 136, 255)        # 橘底（BGR）
    box[4:10, 3:9, :3] = 255             # 白數字
    return box


class FakeOcr:
    """照呼叫順序回傳固定答案，並記錄被呼叫幾次。"""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, image, scale=4, bright_text=True) -> str:
        self.calls += 1
        if not self.answers:
            return ""
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


class TestFingerprint(unittest.TestCase):
    def test_same_pixels_same_hash(self):
        a = testfont.render_bgra("123")
        b = testfont.render_bgra("123")
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_different_pixels_different_hash(self):
        self.assertNotEqual(
            fingerprint(testfont.render_bgra("123")),
            fingerprint(testfont.render_bgra("124")),
        )

    def test_alpha_channel_is_ignored(self):
        a = testfont.render_bgra("123")
        b = a.copy()
        b[..., 3] = 7
        self.assertEqual(fingerprint(a), fingerprint(b))


class TestOrangeMask(unittest.TestCase):
    def test_matches_orange_only(self):
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        image[0, 0, :3] = (30, 150, 240)    # BGR 橘
        image[0, 1, :3] = (240, 240, 240)   # 白
        image[0, 2, :3] = (200, 60, 30)     # 藍
        image[0, 3, :3] = (20, 20, 20)      # 黑
        mask = find_orange_mask(image)
        self.assertEqual(list(mask[0]), [True, False, False, False])


class TestLevelReader(unittest.TestCase):
    def test_cannot_read_before_learning(self):
        reader = LevelReader()
        self.assertIsNone(reader.read(level_zone("44")))

    def test_learns_from_a_known_level(self):
        reader = LevelReader()
        learned = reader.teach(level_zone("44"), 44)
        self.assertEqual(learned, 1)          # 兩個 4 是同一個字形
        self.assertEqual(reader.read(level_zone("44")), 44)
        self.assertEqual(reader.known_digits, "4")

    def test_learns_new_digits_on_level_up(self):
        """只要教過一次，之後每次升級都能自動補上新數字。"""
        reader = LevelReader()
        reader.teach(level_zone("44"), 44)
        reader.teach(level_zone("45"), 45)     # 升級：新等級必定是舊的 +1
        self.assertEqual(reader.known_digits, "45")
        self.assertEqual(reader.read(level_zone("54")), 54)

    def test_rejects_mismatched_digit_count(self):
        reader = LevelReader()
        self.assertEqual(reader.teach(level_zone("44"), 7), 0)

    def test_rejects_out_of_range(self):
        reader = LevelReader()
        for char in "0123456789":
            reader.teach(level_zone(char), int(char) or 1)
        reader.templates.clear()
        for char in "0123456789":
            reader.templates.add(char, testfont.glyph_mask(char))
        self.assertIsNone(reader.read(level_zone("999")))   # 超出 1-300

    def test_unknown_glyph_gives_none_not_a_guess(self):
        reader = LevelReader()
        reader.teach(level_zone("4"), 4)
        self.assertIsNone(reader.read(level_zone("7")))


class TestMapWatcher(unittest.TestCase):
    def test_requires_consecutive_stability(self):
        watcher = MapWatcher(stable_samples=3)
        self.assertFalse(watcher.feed("aaa", None))
        self.assertFalse(watcher.feed("aaa", None))
        self.assertTrue(watcher.feed("aaa", None))
        self.assertEqual(watcher.map_id, "aaa")

    def test_flapping_never_commits(self):
        """地圖區域一直在變（小地圖被收合、或框到遊戲畫面）就永遠不會認定。"""
        watcher = MapWatcher(stable_samples=3)
        for index in range(30):
            self.assertFalse(watcher.feed(f"id{index}", None))
        self.assertEqual(watcher.map_id, "")

    def test_map_change_is_reported_once(self):
        watcher = MapWatcher(stable_samples=2)
        watcher.feed("aaa", None)
        self.assertTrue(watcher.feed("aaa", None))
        self.assertFalse(watcher.feed("aaa", None))   # 同一張圖不再回報
        watcher.feed("bbb", None)
        self.assertTrue(watcher.feed("bbb", None))
        self.assertEqual(watcher.map_id, "bbb")

    def test_empty_id_resets_candidate(self):
        watcher = MapWatcher(stable_samples=2)
        watcher.feed("aaa", None)
        self.assertFalse(watcher.feed("", None))
        self.assertFalse(watcher.feed("aaa", None))   # 連續性被打斷，要重新累積
        self.assertTrue(watcher.feed("aaa", None))


# 小地圖名稱區的實際底色（BGR），實機量到的。
PANEL_BGR = (204, 187, 153)


def make_panel(
    text: str = "1234",
    with_icon: bool = True,
    panel_bgr: tuple[int, int, int] = PANEL_BGR,
    panel_rect: tuple[int, int, int, int] = (18, 20, 132, 64),
) -> np.ndarray:
    """合成一塊小地圖面板：鋼藍底、左邊一個正方形圖示、右邊兩行深色文字。"""
    screen = np.zeros((200, 300, 4), dtype=np.uint8)
    rng = np.random.default_rng(11)
    screen[..., :3] = rng.integers(0, 80, size=(200, 300, 3), dtype=np.uint8)
    screen[..., 3] = 255

    left, top, right, bottom = panel_rect
    screen[top:bottom, left:right, :3] = panel_bgr

    if with_icon:
        # 圖示：接近正方形、幾乎佔滿整條名稱區的高度
        size = bottom - top - 6
        screen[top + 3 : top + 3 + size, left + 3 : left + 3 + size, :3] = 25

    if text:
        mask = testfont.render_mask(text, padding=0)
        mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
        h, w = mask.shape
        ty = top + (bottom - top - h) // 2
        tx = left + (bottom - top) + 6
        region = screen[ty : ty + h, tx : tx + w, :3]
        region[mask] = 30
    return screen


class TestMapZoneValidation(unittest.TestCase):
    def test_rejects_dark_game_scene(self):
        """小地圖被收合時那塊是遊戲畫面 -> 沒有面板色，不該誤判。"""
        scene = np.zeros((200, 300, 4), dtype=np.uint8)
        rng = np.random.default_rng(3)
        scene[..., :3] = rng.integers(0, 70, size=(200, 300, 3), dtype=np.uint8)
        self.assertIsNone(locate_map_zone(scene))

    def test_rejects_neutral_grey_panel(self):
        """中性灰不是小地圖的底色。要求偏藍，才不會被遊戲裡的灰色介面騙到。"""
        screen = make_panel(panel_bgr=(205, 205, 205))
        self.assertIsNone(locate_map_zone(screen))

    def test_rejects_panel_without_text(self):
        screen = make_panel(text="", with_icon=False)
        self.assertIsNone(locate_map_zone(screen))

    def test_finds_text_and_drops_the_icon(self):
        """左邊那個正方形地圖圖示必須被丟掉，只留文字。"""
        screen = make_panel(text="1234", with_icon=True)
        rect = locate_map_zone(screen)
        self.assertIsNotNone(rect)
        left, top, right, bottom = rect
        icon_right = 18 + 3 + (64 - 20 - 6)
        self.assertGreater(left, icon_right, "圖示沒有被排除")
        self.assertLess(right - left, 70)

    def test_works_without_icon(self):
        rect = locate_map_zone(make_panel(text="1234", with_icon=False))
        self.assertIsNotNone(rect)

    def test_finds_panel_wherever_it_is_in_the_corner(self):
        """面板會上下移動（實測同一個 session 就從 y=20 跑到 y=105），不能寫死位置。"""
        high = locate_map_zone(make_panel(panel_rect=(18, 14, 132, 58)))
        low = locate_map_zone(make_panel(panel_rect=(18, 56, 132, 100)))
        self.assertIsNotNone(high)
        self.assertIsNotNone(low)
        self.assertGreater(low[1], high[1] + 30)
        # 同樣的內容在不同位置，裁出來的大小要一樣
        self.assertEqual(high[2] - high[0], low[2] - low[0])
        self.assertEqual(high[3] - high[1], low[3] - low[1])

    def test_search_rect_scopes_the_hunt(self):
        """背包與角色資料視窗用的是**同一種藍底**，顏色分不開。

        真正的防線是由 panels 模組先定位出「小地圖」面板，再把範圍傳進來。
        曾經改用「只找畫面左上角」當權宜，但玩家把小地圖拖走之後就失效了。
        """
        left_panel = make_panel(panel_rect=(18, 20, 118, 64))
        right_panel = make_panel(panel_rect=(170, 20, 270, 64))
        screen = np.maximum(left_panel, right_panel)

        only_left = locate_map_zone(screen, (0, 0, 140, 90))
        only_right = locate_map_zone(screen, (150, 0, 300, 90))
        self.assertIsNotNone(only_left)
        self.assertIsNotNone(only_right)
        self.assertLess(only_left[0], 140)
        self.assertGreater(only_right[0], 150)

    def test_no_rect_means_whole_frame(self):
        """沒給範圍就掃整張 —— 所以呼叫端有責任先把面板定位出來。"""
        screen = make_panel(panel_rect=(170, 20, 270, 64))
        self.assertIsNotNone(locate_map_zone(screen))


if __name__ == "__main__":
    unittest.main()


class TestLevelByOcr(unittest.TestCase):
    """等級方塊的 OCR 後備。

    為什麼需要它：模板比對精確但要先學過字形，而學習只發生在升級的當下。玩家
    中途才開程式時，只要目前等級用到沒學過的數字就整個讀不到（實測模板檔裡
    只有「4」，所以 44 讀得到、45 讀不到）。
    """

    def setUp(self) -> None:
        self._real = identity_module.ocr.recognize
        self.addCleanup(lambda: setattr(identity_module.ocr, "recognize", self._real))

    def use(self, fake: FakeOcr) -> FakeOcr:
        identity_module.ocr.recognize = fake
        return fake

    def test_agreeing_scales_are_accepted(self):
        self.use(FakeOcr("44"))
        self.assertEqual(read_level_by_ocr(orange_box()), 44)

    def test_disagreeing_scales_are_refused(self):
        """方塊只有 22x13，單一倍率不可靠 —— 有分歧就寧可留空。"""
        self.use(FakeOcr("44", "41", "44"))
        self.assertIsNone(read_level_by_ocr(orange_box()))

    def test_silence_is_not_an_answer(self):
        self.use(FakeOcr(""))
        self.assertIsNone(read_level_by_ocr(orange_box()))

    def test_letter_lookalikes_become_digits(self):
        """等級一定是數字，所以 O/l/S 這種誤認可以直接換回去。"""
        self.use(FakeOcr("lO"))
        self.assertEqual(read_level_by_ocr(orange_box()), 10)

    def test_three_digits(self):
        self.use(FakeOcr("120"))
        self.assertEqual(read_level_by_ocr(orange_box()), 120)

    def test_out_of_range_is_refused(self):
        self.use(FakeOcr("999"))
        self.assertIsNone(read_level_by_ocr(orange_box()))

    def test_non_numeric_is_refused(self):
        self.use(FakeOcr("4X"))
        self.assertIsNone(read_level_by_ocr(orange_box()))

    def test_no_image_is_not_an_error(self):
        self.use(FakeOcr("44"))
        self.assertIsNone(read_level_by_ocr(None))


class TestLevelReaderFallback(unittest.TestCase):
    def setUp(self) -> None:
        self._real = identity_module.ocr.recognize
        self.addCleanup(lambda: setattr(identity_module.ocr, "recognize", self._real))

    def test_falls_back_to_ocr_when_the_glyph_is_unknown(self):
        identity_module.ocr.recognize = FakeOcr("44")
        reader = LevelReader()
        self.assertEqual(reader.read(level_zone("44", orange_box())), 44)

    def test_ocr_teaches_the_templates(self):
        """OCR 讀出來就順手把字形學起來，下次走精確又快的模板路。"""
        fake = FakeOcr("44")
        identity_module.ocr.recognize = fake
        reader = LevelReader()
        reader.read(level_zone("44", orange_box()))
        self.assertEqual(reader.known_digits, "4")

        before = fake.calls
        self.assertEqual(reader.read(level_zone("44", orange_box())), 44)
        self.assertEqual(fake.calls, before, "學過之後不該再呼叫 OCR")

    def test_does_not_teach_when_the_digit_count_disagrees(self):
        """切出兩個字形卻讀成三位數 —— 教下去會把錯的字形永久記起來。"""
        identity_module.ocr.recognize = FakeOcr("123")
        reader = LevelReader()
        self.assertEqual(reader.read(level_zone("44", orange_box())), 123)
        self.assertEqual(reader.known_digits, "")

    def test_repeated_failures_stop_calling_ocr(self):
        """OCR 這條路要 66ms；字形沒變的話重試的結果也不會變。"""
        fake = FakeOcr("")
        identity_module.ocr.recognize = fake
        reader = LevelReader()
        zone = level_zone("44", orange_box())
        for _ in range(identity_module.OCR_RETRY_LIMIT):
            self.assertIsNone(reader.read(zone))
        settled = fake.calls
        for _ in range(5):
            self.assertIsNone(reader.read(zone))
        self.assertEqual(fake.calls, settled, "放棄之後不該再呼叫 OCR")

    def test_templates_take_priority_over_ocr(self):
        fake = FakeOcr("99")
        identity_module.ocr.recognize = fake
        reader = LevelReader()
        reader.teach(level_zone("44"), 44)
        self.assertEqual(reader.read(level_zone("44", orange_box())), 44)
        self.assertEqual(fake.calls, 0)
