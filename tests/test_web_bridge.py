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
        return bridge.WebSession(templates=self.templates)

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
        self.assertEqual(out["exp_abs"], 623456)
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
        self.assertEqual(out["exp_abs"], 623456)

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
        session = bridge.WebSession(templates=self.templates)
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
