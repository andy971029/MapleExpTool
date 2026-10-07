"""TrackingSession 的協調層測試。

core 與 vision 各自有紮實的測試，但「把它們串起來」的這一層之前完全沒有 ——
身分掃描曾經因為一個取餘數的閘門寫錯而整個停掉（等級、角色、地圖全部讀不到），
所有單元測試照樣全綠。這裡用假的擷取器與辨識器驅動 ``tick()``，驗證管線真的有
接起來。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401
import numpy as np
from helpers import miss, reading

from mapleexp.config import Config
from mapleexp.core import session as session_module
from mapleexp.core.session import REACQUIRE_AFTER_MISSES, TrackingSession
from mapleexp.vision.identity import Identity
from mapleexp.vision.templates import TemplateSet
from mapleexp.win32.capture import Frame
from mapleexp.win32.windows import WindowInfo


def blank_frame(height: int = 40, width: int = 60) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.uint8)


class FakeCapturer:
    """永遠給一張黑畫面。建構子收任何參數，才能頂替 WindowCapturer。"""

    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    def capture_client(self) -> Frame:
        return Frame(pixels=blank_frame(), backend="fake")

    def close(self) -> None:
        self.closed = True


class FakeReader:
    """照順序吐出預先排好的取樣，用完就一直回失敗。"""

    def __init__(self, *_args, **_kwargs) -> None:
        self.readings = []

    def read(self):
        return self.readings.pop(0) if self.readings else miss(0.0)


def window(hwnd: int) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd, title="新楓之谷", process="x.exe",
        client_width=60, client_height=40, minimized=False,
    )


def identity_with_map(map_id: str, name_image=None):
    def fake_scan(frame, level_reader=None, exp_rect=None, map_rect=None):
        return Identity(map_id=map_id, name_image=name_image)
    return fake_scan


def name_image(seed: int) -> np.ndarray:
    """兩行亮字、中間隔一段黑（_text_bands 會切成兩個 band）。

    ``seed`` 只是讓不同角色的像素不一樣 —— 像素指紋沒變的話不會重跑 OCR。
    """
    image = blank_frame(24, 60)
    image[3:9, 5:40, :3] = 255
    image[14:20, 5:50, :3] = 255
    image[0, 0, 0] = seed
    return image


class SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        # 資料目錄指到暫存，才不會讀到使用者機器上學過的等級字形。
        self._env = mock.patch.dict(os.environ, {"MAPLEEXP_HOME": str(tmp)})
        self._env.start()
        self.session = TrackingSession(
            Config(),
            templates_path=tmp / "digits.json",
            exp_table_path=tmp / "exp_table.json",
        )
        # 跳過 setup()：不找視窗、不自動定位，直接把假元件塞進去。
        self.session.window = window(1)
        self.session.capturer = FakeCapturer()
        self.session.reader = FakeReader()
        self.session._templates = TemplateSet()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()


class TestIdentityPipeline(SessionTestCase):
    def test_identity_is_scanned_on_every_tick(self):
        """這條就是抓到「% 1 == 1 永遠為假」那個 bug 的測試。"""
        self.session.reader.readings = [
            reading(0, 1000, need=10_000),
            reading(1, 1100, need=10_000),
            miss(2),
        ]
        calls = []

        def counting_scan(frame, level_reader=None, exp_rect=None, map_rect=None):
            calls.append(frame.shape)
            return Identity()

        with mock.patch.object(session_module, "scan", counting_scan):
            for _ in range(3):
                self.session.tick()
        # 讀得到、讀不到都要掃 —— 地圖與角色不依賴經驗值欄位。
        self.assertEqual(len(calls), 3)

    def test_stable_map_fingerprint_opens_a_segment(self):
        self.session.reader.readings = [
            reading(i, 1000 + 100 * i, need=10_000) for i in range(4)
        ]
        with mock.patch.object(session_module, "scan", identity_with_map("map-a")):
            for _ in range(2):
                self.session.tick()
            self.assertEqual(self.session.current_map_id, "")   # 還沒穩定三次
            self.session.tick()
            self.assertEqual(self.session.current_map_id, "map-a")
            self.session.tick()

        self.assertEqual(len(self.session.segments), 1)
        segment = self.session.segments[0]
        self.assertEqual(segment.map_id, "map-a")
        self.assertGreater(segment.exp, 0)
        self.assertGreater(segment.active_sec, 0)
        self.assertIsNotNone(segment.start_pct)
        self.assertGreater(segment.end_pct, segment.start_pct)
        self.assertAlmostEqual(segment.pct_gained, segment.end_pct - segment.start_pct)

    def test_segment_peak_rate_is_computed_online(self):
        """峰值不靠事後翻取樣算，追蹤時就用段內 5 分鐘窗的最大斜率記下來。"""
        # 前 60 秒每秒 +100、之後 400 秒每秒 +300：最後一個 5 分鐘窗整段都在 +300，
        # 峰值應該接近 300/s；平均則被前段拉低。
        exps = [1000 + 100 * i for i in range(60)]
        exps += [exps[-1] + 300 * (i + 1) for i in range(400)]
        self.session.reader.readings = [
            reading(i, exp, need=1_000_000) for i, exp in enumerate(exps)
        ]
        with mock.patch.object(session_module, "scan", identity_with_map("map-a")):
            for _ in range(len(exps)):
                self.session.tick()
        segment = self.session.segments[0]
        self.assertIsNotNone(segment.peak_exp_per_hour)
        self.assertGreater(segment.peak_exp_per_hour, 290 * 3600)
        self.assertLess(segment.peak_exp_per_hour, 310 * 3600)
        self.assertLess(segment.exp_per_hour, segment.peak_exp_per_hour)

    def test_levelup_teaches_the_new_glyph_from_the_pre_levelup_level(self):
        """升級後模板認不出新字形時，用「掃描前的等級 + 1」教。

        不能用掃描後的 level_reader.level：那一格的掃描可能已經讀出新等級了，
        再 +1 就把錯的數字永久記進模板。
        """
        from mapleexp.vision.identity import LevelZone

        self.session.level_reader.level = 44
        zone = LevelZone(rect=(0, 0, 10, 10), digits=[np.ones((5, 3), dtype=bool)] * 2)
        taught = []
        self.session.level_reader.teach = lambda z, level: taught.append(level) or 1
        self.session.reader.readings = [
            reading(0, 9700, need=10_000),
            reading(1, 9800, need=10_000),
            reading(2, 150, need=20_000),     # 歸零：升級，等確認
            reading(3, 350, need=20_000),     # 確認 -> levelup 事件
            reading(4, 450, need=20_000),     # 下一次掃描：教字形
        ]
        with mock.patch.object(
            session_module, "scan", lambda *a, **k: Identity(zone=zone, level=None)
        ):
            for _ in range(5):
                self.session.tick()
        self.assertEqual(taught, [45])

    def test_level_from_level_reader_is_attached_to_reading(self):
        self.session.level_reader.level = 44
        self.session.reader.readings = [reading(0, 1000, need=10_000)]
        with mock.patch.object(session_module, "scan", lambda *a, **k: Identity()):
            result = self.session.tick()
        self.assertEqual(result.reading.level, 44)
        self.assertEqual(result.snapshot.level, 44)


class TestMinimapPanel(SessionTestCase):
    """小地圖面板的記憶與重找。地圖辨識慢，十之八九是這裡的冷卻被用錯地方。"""

    def _wire(self):
        """假面板搜尋（永遠找得到）+ 只在有面板範圍時才回報地圖的假掃描 + 假時鐘。"""
        from mapleexp.vision.panels import Panel

        state = {"map": "map-a", "clock": 0.0, "searches": 0}

        def refresh(frame):
            state["searches"] += 1
            self.session.panels.panels[session_module.MINIMAP_TITLE] = Panel(
                title="小地圖", raw_title="小地圖", score=1.0,
                title_rect=(0, 0, 60, 12), body_rect=(0, 12, 60, 40),
            )
            return self.session.panels.panels

        def fake_scan(frame, level_reader=None, exp_rect=None, map_rect=None):
            return Identity(map_id=state["map"] if map_rect is not None else "")

        self.session.panels.refresh = refresh
        patches = (
            mock.patch.object(session_module, "scan", fake_scan),
            mock.patch.object(session_module.time, "monotonic", lambda: state["clock"]),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return state

    def _tick(self, state, ok: bool = True) -> None:
        state["clock"] += 1.0
        self.session.reader.readings = [
            reading(state["clock"], 1000, need=10_000) if ok else miss(state["clock"])
        ]
        self.session.tick()

    def test_loading_screen_does_not_lose_the_panel(self):
        """換圖：讀取畫面那幾格狀態列讀不到，面板要留著，進新圖後 3 格內就認得。"""
        state = self._wire()
        for _ in range(4):
            self._tick(state)
        self.assertEqual(self.session.current_map_id, "map-a")
        self.assertEqual(state["searches"], 1)

        state["map"] = ""                     # 讀取畫面：小地圖不在
        for _ in range(4):
            self._tick(state, ok=False)
        self.assertIsNotNone(self.session.panels.get(session_module.MINIMAP_TITLE))

        state["map"] = "map-b"
        for _ in range(3):
            self._tick(state)
        self.assertEqual(self.session.current_map_id, "map-b")
        self.assertEqual(state["searches"], 1)   # 根本不需要重找

    def test_moved_panel_is_relocated_without_waiting_for_the_cooldown(self):
        """面板真的被搬走：連續幾格讀不到才忘掉，忘掉後下一格就重找。"""
        state = self._wire()
        for _ in range(4):
            self._tick(state)
        self.assertEqual(state["searches"], 1)

        state["map"] = ""                     # 狀態列正常，但面板範圍裡沒有地圖名
        for _ in range(session_module.MAP_LOST_TICKS - 1):
            self._tick(state)
        self.assertIsNotNone(self.session.panels.get(session_module.MINIMAP_TITLE))
        self._tick(state)                     # 第 N 格：忘掉
        self.assertIsNone(self.session.panels.get(session_module.MINIMAP_TITLE))

        state["map"] = "map-a"
        self._tick(state)                     # 下一格就重找，不用等 20 秒
        self.assertEqual(state["searches"], 2)
        self.assertIsNotNone(self.session.panels.get(session_module.MINIMAP_TITLE))

    def test_failed_search_is_throttled(self):
        state = self._wire()
        found = {"yes": False}
        original = self.session.panels.refresh

        def flaky_refresh(frame):
            if found["yes"]:
                return original(frame)       # original 自己會計數
            state["searches"] += 1
            return {}

        self.session.panels.refresh = flaky_refresh
        for _ in range(5):
            self._tick(state)
        self.assertEqual(state["searches"], 1)        # 失敗後 20 秒內不再試
        state["clock"] += session_module.PANEL_REFRESH_INTERVAL
        found["yes"] = True
        self._tick(state)
        self.assertEqual(state["searches"], 2)


class TestDeleteSegment(SessionTestCase):
    def test_deleted_segment_is_dropped_and_deducted_from_totals(self):
        """結束視窗按垃圾桶：那一段不寫入，而且這一場的總計要跟著扣掉。"""
        self.session.reader.readings = [
            reading(i, 1000 + 100 * i, need=10_000) for i in range(6)
        ]
        with mock.patch.object(session_module, "scan", identity_with_map("map-a")):
            for _ in range(6):
                self.session.tick()
        (segment,) = self.session.segments
        self.assertGreater(segment.exp, 0)

        self.assertTrue(self.session.delete_segment(segment))
        self.assertEqual(self.session.segments, [])
        self.assertFalse(self.session.delete_segment(segment))   # 已經不在了

        totals = self.session._session_snapshot()
        raw = self.session.tracker.snapshot()
        self.assertEqual(totals.cum_net, raw.cum_net - segment.exp)
        self.assertAlmostEqual(totals.active_sec, raw.active_sec - segment.active_sec)

    def test_restart_clears_deductions(self):
        self.session._deducted_exp = 500
        self.session.restart()
        self.assertEqual(self.session._deducted_exp, 0)


class TestCharacterAndJob(SessionTestCase):
    def test_job_is_matched_against_vocabulary(self):
        # 兩行亮字、中間隔一段黑：_text_bands 會切成兩個 band。
        image = blank_frame(24, 60)
        image[3:9, 5:40, :3] = 255
        image[14:20, 5:50, :3] = 255
        answers = iter(["搶騎兵", "尼可拉絲麻吉"])
        with mock.patch.object(
            session_module.ocr, "recognize_best", lambda img, **k: next(answers)
        ):
            self.session._read_character(image)
        self.assertEqual(self.session.job, "槍騎兵")
        self.assertEqual(self.session.character, "尼可拉絲麻吉")

    def test_unreadable_image_keeps_previous_names(self):
        self.session.job, self.session.character = "獵人", "某人"
        with mock.patch.object(session_module.ocr, "recognize_best", lambda img, **k: ""):
            self.session._read_character(blank_frame(24, 60))
        self.assertEqual((self.session.job, self.session.character), ("獵人", "某人"))

    def test_switching_character_closes_the_segment_and_rebases(self):
        """換角色：舊段結束、新段掛在新角色名下，兩隻角色的經驗差不算死亡。"""
        answers = iter(["獵人", "甲", "刺客", "乙"])
        current = {"image": name_image(1)}

        def fake_scan(frame, level_reader=None, exp_rect=None, map_rect=None):
            return Identity(map_id="map-a", name_image=current["image"])

        self.session.reader.readings = [
            reading(0, 1000, need=10_000),
            reading(1, 1100, need=10_000),
            reading(2, 1200, need=10_000),
            reading(3, 50, need=5_000),        # 另一隻角色
            reading(4, 150, need=5_000),
        ]
        with mock.patch.object(session_module, "scan", fake_scan), \
             mock.patch.object(session_module, "CHARACTER_RECHECK_SEC", 0.0), \
             mock.patch.object(session_module.ocr, "recognize_best",
                               lambda img, **k: next(answers)):
            for _ in range(3):
                self.session.tick()
            self.assertEqual(self.session.character, "甲")
            current["image"] = name_image(2)   # 狀態列上的名字變了
            result = self.session.tick()
            self.session.tick()

        self.assertEqual(self.session.character, "乙")
        self.assertEqual(self.session.job, "刺客")
        self.assertIn("character", [e.kind for e in result.events])
        self.assertIn("rebase", [e.kind for e in result.events])

        first, second = self.session.segments
        self.assertEqual((first.character, first.job), ("甲", "獵人"))
        self.assertIsNotNone(first.ended_at)
        # 地圖要穩定三格才開段，第 2 格的 +100 不歸任何段；段內只有第 3 格的 +100。
        self.assertEqual(first.exp, 100)
        self.assertEqual((second.character, second.job), ("乙", "刺客"))
        self.assertIsNone(second.ended_at)
        self.assertEqual(second.exp, 100)

        snapshot = self.session.tracker.snapshot()
        self.assertEqual(snapshot.deaths, 0)
        self.assertEqual(snapshot.cum_net, 300)

    def test_resuming_after_a_pause_rechecks_the_character_immediately(self):
        """畫面消失再回來（選角色畫面）：不等 5 秒，回來那一格就重新確認是誰。"""
        answers = iter(["獵人", "甲", "刺客", "乙"])
        current = {"image": name_image(1)}

        def fake_scan(frame, level_reader=None, exp_rect=None, map_rect=None):
            return Identity(name_image=current["image"])

        self.session.reader.readings = [reading(0, 1000, need=10_000)] + [
            miss(i) for i in range(1, 5)
        ] + [reading(5, 50, need=5_000)]
        with mock.patch.object(session_module, "scan", fake_scan), \
             mock.patch.object(session_module.ocr, "recognize_best",
                               lambda img, **k: next(answers)):
            self.session.tick()
            self.assertEqual(self.session.character, "甲")
            for _ in range(4):
                self.session.tick()                     # 連續失敗 -> 暫停
            current["image"] = name_image(2)
            # 距離上次確認不到 5 秒，但追蹤器處於暫停、畫面剛回來 -> 立刻重讀
            self.session.tick()
        self.assertEqual(self.session.character, "乙")
        self.assertEqual(self.session.tracker.snapshot().deaths, 0)


class TestWindowReacquire(SessionTestCase):
    def test_reacquire_counts_the_current_miss(self):
        """第 N 次失敗的那一格就該重新找視窗，不是第 N+1 次。"""
        self.session.reader.readings = [miss(i) for i in range(10)]
        patches = (
            mock.patch.object(session_module, "window_exists", return_value=False),
            mock.patch.object(session_module, "find_window", return_value=window(2)),
            mock.patch.object(session_module, "WindowCapturer", FakeCapturer),
            mock.patch.object(session_module, "StatusReader", FakeReader),
            mock.patch.object(session_module, "scan", lambda *a, **k: Identity()),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        for _ in range(REACQUIRE_AFTER_MISSES - 1):
            self.session.tick()
        self.assertEqual(self.session.window.hwnd, 1)
        self.session.tick()
        self.assertEqual(self.session.window.hwnd, 2)
        self.assertEqual(self.session.last_error, "遊戲視窗已重新連接")

    def test_no_reacquire_while_window_still_exists(self):
        self.session.reader.readings = [miss(i) for i in range(10)]
        with mock.patch.object(session_module, "window_exists", return_value=True), \
             mock.patch.object(session_module, "find_window", return_value=window(2)), \
             mock.patch.object(session_module, "scan", lambda *a, **k: Identity()):
            for _ in range(REACQUIRE_AFTER_MISSES + 1):
                self.session.tick()
        self.assertEqual(self.session.window.hwnd, 1)


if __name__ == "__main__":
    unittest.main()
