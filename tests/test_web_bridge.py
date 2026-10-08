"""瀏覽器版膠水層（web/bridge.py）。

bridge 在 Pyodide 裡跑，但它的邏輯跟平台無關：餵一張合成畫面進去，應該跟桌面版
一樣找得到經驗值欄位、讀得出數字。這裡在 CPython 下直接匯入它驗證，省掉開瀏覽器。\n"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401
import numpy as np

from mapleexp import testfont
from test_locate import make_screen

WEB = Path(__file__).resolve().parents[1] / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

import bridge  # noqa: E402


def rgba_bytes(screen_bgra: np.ndarray) -> bytes:
    """模擬 canvas getImageData：RGBA 順序的連續位元組。"""
    return np.ascontiguousarray(screen_bgra[..., [2, 1, 0, 3]]).tobytes()


class TestWebSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = testfont.build_template_set()

    def make_session(self) -> bridge.WebSession:
        return bridge.WebSession(templates=self.templates, require_bracket=False)

    def feed_over_time(self, session: bridge.WebSession, texts: list[str]) -> dict:
        """速率要有時間差才算得出來；合成畫面瞬間餵完，所以讓時鐘每格走 1 秒。"""
        clock = iter(range(1000, 1000 + 10 * len(texts)))
        out: dict = {}
        with mock.patch("mapleexp.vision.reader.time.perf_counter", lambda: float(next(clock))):
            for text in texts:
                out = self.feed(session, text)
        return out

    def feed(self, session: bridge.WebSession, text: str) -> dict:
        screen = make_screen(text)
        h, w = screen.shape[:2]
        session.feed_frame(rgba_bytes(screen), w, h)
        return json.loads(session.tick())

    def test_rgba_from_canvas_becomes_bgra(self):
        """canvas 給 RGBA、辨識層要 BGRA；換錯通道整條管線都會讀錯色。"""
        capturer = bridge.FrameCapturer()
        pixel = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)    # R G B A
        capturer.set_frame(pixel.tobytes(), 1, 1)
        self.assertEqual(capturer.capture_client().pixels[0, 0].tolist(), [30, 20, 10, 255])

    def test_first_tick_locates_and_reads(self):
        session = self.make_session()
        out = self.feed(session, "623456[12.34%]")
        self.assertTrue(out["located"], out["message"])
        self.assertEqual(out["raw_exp"], "623456[12.34%]")
        self.assertEqual(out["exp_abs"], 623456, out["message"])
        self.assertIsNotNone(out["rect"])
        self.assertEqual(out["frame_size"], [make_screen("1").shape[1], make_screen("1").shape[0]])

    def test_gain_across_ticks_accumulates(self):
        session = self.make_session()
        out = self.feed_over_time(session, ["623456[12.34%]", "623956[12.35%]", "624456[12.36%]"])
        self.assertEqual(out["exp_abs"], 624456)
        self.assertEqual(out["state_key"], "active")

    def test_five_stats(self):
        session = self.make_session()
        out = self.feed_over_time(session, ["623456[12.34%]", "623956[12.35%]", "624456[12.36%]"])
        stats = out["stats"]
        self.assertTrue(stats["recent_valid"])
        # 第一次增加前的閒置間隔不算活躍時間，所以窗內累計可能少於總累計，但必須與追蹤器一致
        est = session.tracker.snapshot().rates[bridge.RATE_WINDOW_SEC]
        self.assertEqual(stats["recent"], bridge.format_exp(est.exp_gained))
        self.assertEqual(stats["total"], bridge.format_exp(1000))
        rate = session.tracker.snapshot().rates[bridge.RATE_WINDOW_SEC].exp_per_hour
        self.assertEqual(stats["per_hour"], bridge.format_rate(rate))
        self.assertEqual(stats["per_half_hour"], bridge.format_exp(rate / 2))
        self.assertNotEqual(stats["average"], "--")

    def test_stats_before_any_gain_are_placeholders(self):
        out = self.feed(self.make_session(), "623456[12.34%]")
        self.assertFalse(out["stats"]["recent_valid"])
        self.assertEqual(out["stats"]["recent"], "--")

    def test_payload_lists_every_read_region(self):
        out = self.feed(self.make_session(), "623456[12.34%]")
        self.assertEqual(set(out["rois"]), {"exp", "level", "job", "name", "map"})
        self.assertEqual(out["rois"]["exp"], out["rect"])

    def test_roi_grows_back_after_being_shrunk_by_an_occluder(self):
        """滑鼠遮住欄位時重新定位只會找到一截；遮蔽移開後 ROI 要自己長回來。"""
        session = self.make_session()
        self.feed(session, "623456[12.34%]")
        full = session.cfg.reader.exp_roi
        session.cfg.reader.exp_roi = bridge.Roi(full.anchor, full.dx + 15, full.dy, full.w - 15, full.h)
        session._widen_checked_at = 0.0
        self.feed(session, "623456[12.34%]")
        grown = session.cfg.reader.exp_roi
        self.assertGreaterEqual(grown.w, full.w - 2)
        self.assertEqual(session._current_rect()[2] - session._current_rect()[0], grown.w)

    def test_widen_check_never_shrinks_the_roi(self):
        session = self.make_session()
        self.feed(session, "623456[12.34%]")
        before = session.cfg.reader.exp_roi
        session._widen_checked_at = 0.0
        self.feed(session, "623456[12.34%]")
        self.assertEqual(session.cfg.reader.exp_roi, before)

    def test_missing_closing_bracket_pauses_counting(self):
        """綠色的 ] 看不到＝欄位被擋住、資料不齊全，這格不能記進統計。"""
        session = bridge.WebSession(templates=self.templates)      # 預設要求括號
        screen = make_screen("623456[12.34%]")
        h, w = screen.shape[:2]
        session.feed_frame(rgba_bytes(screen), w, h)
        out = json.loads(session.tick())
        self.assertTrue(session.occluded)
        self.assertEqual(out["misses"], 1)
        self.assertIn("]", out["message"])

        left, top, right, bottom = session.rect
        tail = screen[top:bottom, right - 6 : right, :3]
        tail[tail.max(axis=2) > 200] = (60, 220, 90)       # 把 ] 本身染綠（BGR），不破壞字形
        session.feed_frame(rgba_bytes(screen), w, h)
        out = json.loads(session.tick())
        self.assertFalse(session.occluded, out["message"])
        self.assertEqual(out["exp_abs"], 623456, out["message"])

    def test_map_name_is_found_via_minimap_title_then_read(self):
        """小地圖標題列 OCR 成「小地圖」→ 內容區定位 → 指紋穩定換圖 → 名稱 OCR。"""
        from test_identity import make_panel

        screen = make_screen(clutter=False, extra=None)
        screen[:80, :150] = make_panel()[:80, :150]
        screen[4:18, 18:132, :3] = 255                      # 純白標題列
        h, w = screen.shape[:2]
        session = self.make_session()
        session.set_ocr_ready(True)

        def tick() -> dict:
            session.feed_frame(rgba_bytes(screen), w, h)
            return json.loads(session.tick())

        out = tick()
        requests = {req["id"]: req for req in out["ocr"]}
        self.assertIn("title:minimap", requests)
        session.ocr_result("title:minimap", json.dumps([["小地", "小地圖"]] * len(requests["title:minimap"]["images"])))

        requests = {}
        for _ in range(4):                                   # 指紋要連續穩定 3 格才算換圖
            out = tick()
            requests.update({req["id"]: req for req in out["ocr"]})
        self.assertIsNotNone(out["rois"]["map"])
        self.assertTrue(out["map_id"])
        self.assertIn(f"map:{out['map_id']}", requests)

        session.ocr_result(f"map:{out['map_id']}", json.dumps([["戰火之地"], ["沼澤地III"]]))
        self.assertEqual(tick()["map_name"], "戰火之地 · 沼澤地III")

    def test_map_name_snaps_to_the_vocabulary_snapshot(self):
        """OCR 缺字、羅馬數字讀成「川」，都該被清單修回官方名稱。"""
        from test_identity import make_panel

        screen = make_screen(clutter=False, extra=None)
        screen[:80, :150] = make_panel()[:80, :150]
        screen[4:18, 18:132, :3] = 255
        h, w = screen.shape[:2]
        session = self.make_session()
        self.assertEqual(session.set_map_vocab(json.dumps([
            ["1", "戰火之地", "沼澤地Ⅰ"], ["2", "戰火之地", "沼澤地Ⅲ"], ["3", "維多利亞港", "碼頭"],
        ], ensure_ascii=False)), 3)
        session.set_ocr_ready(True)
        out = {}
        for i in range(6):
            session.feed_frame(rgba_bytes(screen), w, h)
            out = json.loads(session.tick())
            for req in out["ocr"]:
                if req["id"] == "title:minimap":
                    session.ocr_result(req["id"], json.dumps([["小地圖"]] * len(req["images"])))
                elif req["id"].startswith("map:"):
                    session.ocr_result(req["id"], json.dumps([["戰火之地"], ["沼澤地川"]]))
        self.assertEqual(out["map_name"], "戰火之地 沼澤地Ⅲ")

    def test_minimap_is_forgotten_when_zone_disappears(self):
        from test_identity import make_panel

        screen = make_screen(clutter=False, extra=None)
        screen[:80, :150] = make_panel()[:80, :150]
        screen[4:18, 18:132, :3] = 255
        h, w = screen.shape[:2]
        session = self.make_session()
        session.set_ocr_ready(True)
        session.feed_frame(rgba_bytes(screen), w, h)
        out = json.loads(session.tick())
        session.ocr_result("title:minimap", json.dumps([["小地圖"]] * len(out["ocr"][0]["images"])))
        session.feed_frame(rgba_bytes(screen), w, h)
        session.tick()
        self.assertIsNotNone(session._minimap_body)
        screen[:80, :150, :3] = 30                           # 面板被關掉
        for _ in range(bridge.MAP_LOST_TICKS):
            session.feed_frame(rgba_bytes(screen), w, h)
            session.tick()
        self.assertIsNone(session._minimap_body)

    def test_payload_before_any_frame_is_safe(self):
        session = self.make_session()
        out = json.loads(session.tick())
        self.assertFalse(out["located"])
        self.assertIsNone(out["frame_size"])

    def test_reset_keeps_roi(self):
        session = self.make_session()
        self.feed(session, "623456[12.34%]")
        rect = session.rect
        session.reset()
        out = self.feed(session, "623456[12.34%]")
        self.assertEqual(session.rect, rect)
        self.assertEqual(out["exp_abs"], 623456, out["message"])

    def test_rect_follows_window_resize(self):
        """視窗縮放後 reader 靠距底部中央的偏移仍讀得到，回報的欄位位置也要跟著換算，
        否則頁面拿舊座標去裁放大圖會是空白。"""
        session = self.make_session()
        self.feed(session, "623456[12.34%]")
        # 同樣距底部中央 (-30, -30) 的位置，只是畫面變小
        small = make_screen("623956[12.35%]", width=320, height=240, text_x=130, text_y=210)
        session.feed_frame(rgba_bytes(small), 320, 240)
        out = json.loads(session.tick())
        self.assertEqual(out["raw_exp"], "623956[12.35%]")
        left, top, right, bottom = out["rect"]
        self.assertTrue(right <= 320 and bottom <= 240, out["rect"])
        self.assertTrue(120 <= left <= 135, out["rect"])


class TestNumeralBars(unittest.TestCase):
    def _line(self, bars: int) -> np.ndarray:
        # 藍底白字：一個漢字的豎筆（距離 3 px）加上間距 4 px 的直槓。
        image = np.zeros((14, 30, 3), np.uint8)
        image[...] = (160, 130, 90)
        image[2:12, 6] = 255
        for i in range(bars):
            image[2:11, 9 + 4 * i: 11 + 4 * i] = 255
        return image

    def test_counts_bars_but_not_the_preceding_stroke(self):
        for bars in (1, 2, 3):
            self.assertEqual(bridge.trailing_numeral_bars(self._line(bars), (0, 14)), bars)

    def test_ocr_single_I_is_corrected_to_the_pixel_count(self):
        from mapleexp.core import mapvocab
        from mapleexp.gamedata import MapRecord, MapVocabulary

        vocab = MapVocabulary(records=tuple(
            MapRecord("107000%d00" % i, "戰火之地", "沼澤地" + n)
            for i, n in enumerate("ⅠⅡⅢ")
        ))
        guess = mapvocab.identify("戰火之地沼澤地I", vocab)
        self.assertEqual(guess.name, "沼澤地Ⅰ")
        self.assertEqual(bridge._fix_numeral(guess, 3, vocab), "戰火之地 沼澤地Ⅲ")
        self.assertIsNone(bridge._fix_numeral(guess, 1, vocab))
        self.assertIsNone(bridge._fix_numeral(guess, 0, vocab))


class FakeZone:
    """只帶 bridge 會用到的欄位；真正的字形切割在 test_identity 驗證。"""

    shape_key = "zone-a"
    digits = [object(), object()]


class TestIdentityOcr(unittest.TestCase):
    """JS 端 OCR 交回結果之後，Python 這邊怎麼採信。"""

    @classmethod
    def setUpClass(cls):
        cls.templates = testfont.build_template_set()

    def make_session(self) -> bridge.WebSession:
        session = bridge.WebSession(templates=self.templates, require_bracket=False)
        self.taught: list[int] = []
        session.level_reader.teach = lambda zone, level: self.taught.append(level) or 0
        return session

    def test_level_needs_every_scale_to_agree(self):
        session = self.make_session()
        session._finish_level(FakeZone(), ["45", "45", "", "45"])
        self.assertEqual(session.level, 45)
        self.assertEqual(self.taught, [45])

    def test_level_disagreement_is_rejected_and_counted(self):
        session = self.make_session()
        session._finish_level(FakeZone(), ["45", "46", "45", "45"])
        self.assertIsNone(session.level)
        self.assertEqual(session._level_misses["zone-a"], 1)
        self.assertEqual(self.taught, [])

    def test_level_that_is_not_a_number_is_rejected(self):
        session = self.make_session()
        session._finish_level(FakeZone(), ["45", "4x", "45"])
        self.assertIsNone(session.level)

    def test_level_digit_count_mismatch_is_not_taught(self):
        """切出兩個字形、辨識卻說是三位數：用這個教模板會把錯的字形記起來。"""
        session = self.make_session()
        session._finish_level(FakeZone(), ["145", "145"])
        self.assertEqual(session.level, 145)
        self.assertEqual(self.taught, [])

    def test_character_splits_job_and_name(self):
        session = self.make_session()
        session._finish_character({"key": "k"}, [["獵人\n", "獵人\n"], ["卍 弘法 艾 德 卍\n", "卍 弘法 艾\n"]])
        self.assertEqual(session.job, "獵人")
        self.assertEqual(session.character, "卍弘法艾德卍")

    def test_unmatched_job_does_not_overwrite_a_known_one(self):
        session = self.make_session()
        session.job = "獵人"
        session._finish_character({"key": "k"}, [["亂碼亂碼\n"], ["角色\n"]])
        self.assertEqual(session.job, "獵人")
        self.assertEqual(session.character, "角色")

    def test_engine_failure_keeps_previous_identity(self):
        session = self.make_session()
        session.job, session.character = "獵人", "舊名"
        session._finish_character({"key": "k"}, None)
        self.assertEqual((session.job, session.character), ("獵人", "舊名"))

    def test_no_requests_until_js_engine_is_ready(self):
        session = self.make_session()
        session._request_ocr("x", "digits", ["img"], {})
        self.assertEqual(len(session._take_ocr_outbox()), 1)   # 掛號本身不看 ready
        screen = make_screen("623456[12.34%]")
        h, w = screen.shape[:2]
        session.feed_frame(rgba_bytes(screen), w, h)
        out = json.loads(session.tick())
        self.assertEqual(out["ocr"], [])

    def test_same_request_is_not_queued_twice(self):
        session = self.make_session()
        session._request_ocr("x", "digits", ["img"], {})
        session._request_ocr("x", "digits", ["img"], {})
        self.assertEqual(len(session._take_ocr_outbox()), 1)
        session.ocr_result("x", json.dumps(["1"]))     # 結果回來後才能再掛同一件
        session._request_ocr("x", "digits", ["img"], {})
        self.assertEqual(len(session._take_ocr_outbox()), 1)


if __name__ == "__main__":
    unittest.main()
