"""瀏覽器版膠水層（web/bridge.py）。

bridge 在 Pyodide 裡跑，但它的邏輯跟平台無關：餵一張合成畫面進去，應該跟桌面版
一樣找得到經驗值欄位、讀得出數字。這裡在 CPython 下直接匯入它驗證，省掉開瀏覽器。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

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
        self.feed(session, "623456[12.34%]")
        out = self.feed(session, "623956[12.35%]")
        self.assertEqual(out["exp_abs"], 623956)
        self.assertEqual(out["state_key"], "active")
        self.assertEqual(len(out["rates"]), 3)

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


if __name__ == "__main__":
    unittest.main()
