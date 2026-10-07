"""SQLite 儲存層測試：分段寫入、按日報表、舊資料庫升級。"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import helpers  # noqa: F401

from mapleexp.core.tracker import Event, Snapshot
from mapleexp.storage import Store


def snapshot(**overrides) -> Snapshot:
    base = dict(
        state="active", level=44, exp_abs=1000, exp_pct=10.0, need=10_000,
        need_confidence="derived", remaining=9_000, cum_net=0, cum_gross=0,
        exp_lost=0, active_sec=0.0, wall_sec=0.0, skipped_sec=0.0,
    )
    base.update(overrides)
    return Snapshot(**base)


def segment(**overrides) -> dict:
    base = dict(
        map_id="map-a", character="甲", job="獵人",
        started_at=time.time() - 600, ended_at=time.time(),
        exp=60_000, active_sec=600.0,
        start_level=44, end_level=44, start_pct=10.0, end_pct=24.0,
        peak_exp_per_hour=420_000.0, deaths=0, levelups=0,
    )
    base.update(overrides)
    return base


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "tracker.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestSegmentsAndDailyReport(StoreTestCase):
    def test_segments_round_trip_into_daily_report(self):
        with Store(self.path) as store:
            store.open_session("")
            store.remember_map("map-a", None, "樹林底層")
            store.write_segments(
                [
                    segment(),
                    # 同一隻角色、同一張圖、同一天的第二段：要合併成一列
                    segment(exp=30_000, active_sec=300.0, start_pct=24.0, end_pct=31.0,
                            peak_exp_per_hour=380_000.0),
                    # 另一隻角色，分開列
                    segment(character="乙", job="刺客", exp=10_000, active_sec=100.0,
                            start_pct=50.0, end_pct=52.0, peak_exp_per_hour=None),
                ]
            )
            store.close_session(snapshot(cum_net=100_000, active_sec=1000.0))
            rows = store.daily_report(days=7)

        self.assertEqual(len(rows), 2)
        by_character = {row["character"]: row for row in rows}

        first = by_character["甲"]
        self.assertEqual(first["map_name"], "樹林底層")
        self.assertEqual(first["job"], "獵人")
        self.assertEqual(first["runs"], 2)
        self.assertEqual(first["exp"], 90_000)
        self.assertEqual(first["active_sec"], 900.0)
        self.assertAlmostEqual(first["exp_per_hour"], 360_000.0)
        self.assertEqual(first["peak_exp_per_hour"], 420_000.0)     # 兩段取最大
        self.assertAlmostEqual(first["pct_gained"], 14.0 + 7.0)

        second = by_character["乙"]
        self.assertEqual(second["exp"], 10_000)
        self.assertIsNone(second["peak_exp_per_hour"])
        self.assertAlmostEqual(second["pct_gained"], 2.0)

    def test_list_segments_joins_map_and_label_and_skips_open_sessions(self):
        with Store(self.path) as store:
            store.remember_map("map-a", None, "樹林底層")
            store.open_session("標記甲")
            store.write_segments([
                segment(started_at=time.time() - 600),
                segment(character="乙", exp=5_000, active_sec=100.0, started_at=time.time() - 100),
            ])
            store.close_session(snapshot())
            store.open_session("還沒結束")
            store.write_segments([segment(character="丙")])
            rows = store.list_segments()
        # 最新的在前；還沒結束那場的「丙」不會出現。
        self.assertEqual([r["character"] for r in rows], ["乙", "甲"])
        self.assertEqual(rows[1]["map_name"], "樹林底層")
        self.assertEqual(rows[1]["label"], "標記甲")
        self.assertAlmostEqual(rows[1]["pct_gained"], 14.0)
        self.assertEqual(rows[0]["exp"], 5_000)

    def test_pct_gained_spans_levelups(self):
        with Store(self.path) as store:
            store.open_session("")
            store.write_segments(
                [segment(start_pct=90.0, end_pct=5.0, levelups=1, start_level=44, end_level=45)]
            )
            store.close_session(snapshot())
            (row,) = store.daily_report(days=7)
        self.assertAlmostEqual(row["pct_gained"], 15.0)
        self.assertEqual((row["start_level"], row["end_level"]), (44, 45))

    def test_days_window_and_zero_means_everything(self):
        with Store(self.path) as store:
            store.open_session("")
            store.write_segments([segment(started_at=time.time() - 40 * 86400)])
            store.close_session(snapshot())
            self.assertEqual(store.daily_report(days=30), [])
            self.assertEqual(len(store.daily_report(days=0)), 1)

    def test_discard_removes_segments_and_events(self):
        with Store(self.path) as store:
            store.open_session("")
            store.add_events([Event(kind="start", wall=time.time())])
            store.write_segments([segment()])
            store.discard_session()
            self.assertEqual(store.daily_report(days=0), [])
            self.assertEqual(store.list_sessions(), [])

    def test_prune_incomplete_keeps_sessions_that_had_events(self):
        with Store(self.path) as store:
            store.open_session("沒讀到任何東西就關掉")
            crashed = store.open_session("有讀到東西但沒正常結束")
            store.add_events([Event(kind="start", wall=time.time())])
            store._session_id = None
            self.assertEqual(store.prune_incomplete(), 1)
            remaining = store.list_sessions()
        self.assertEqual([row.id for row in remaining], [crashed])


class TestMigration(StoreTestCase):
    def test_old_database_gets_new_columns_and_loses_samples(self):
        """0.1 的資料庫：segments 沒有角色欄位，而且有一張每秒取樣的 samples 表。"""
        conn = sqlite3.connect(self.path)
        conn.executescript(
            """
            CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL DEFAULT '',
                started_at REAL NOT NULL, ended_at REAL, start_level INTEGER, end_level INTEGER,
                cum_net INTEGER NOT NULL DEFAULT 0, cum_gross INTEGER NOT NULL DEFAULT 0,
                exp_lost INTEGER NOT NULL DEFAULT 0, active_sec REAL NOT NULL DEFAULT 0,
                wall_sec REAL NOT NULL DEFAULT 0, deaths INTEGER NOT NULL DEFAULT 0,
                levelups INTEGER NOT NULL DEFAULT 0, glitches INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0, samples INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE samples (session_id INTEGER NOT NULL, wall REAL NOT NULL,
                active_sec REAL NOT NULL, level INTEGER, exp_abs INTEGER, exp_pct REAL,
                cum_net INTEGER NOT NULL, state TEXT NOT NULL);
            CREATE TABLE segments (session_id INTEGER NOT NULL, map_id TEXT NOT NULL,
                started_at REAL NOT NULL, ended_at REAL, exp INTEGER NOT NULL DEFAULT 0,
                active_sec REAL NOT NULL DEFAULT 0);
            CREATE TABLE maps (map_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
                first_seen REAL NOT NULL, thumbnail BLOB);
            INSERT INTO sessions (started_at, ended_at, active_sec, cum_net) VALUES (1, 2, 600, 1000);
            INSERT INTO samples VALUES (1, 1, 0, 44, 1000, 10.0, 0, 'active');
            INSERT INTO segments VALUES (1, 'map-a', 1, 2, 1000, 600);
            """
        )
        conn.commit()
        conn.close()

        with Store(self.path) as store:
            tables = {
                row[0]
                for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertNotIn("samples", tables)
            columns = {row[1] for row in store._conn.execute("PRAGMA table_info(segments)")}
            self.assertTrue(
                {"character", "job", "peak_exp_per_hour", "start_pct", "levelups"} <= columns
            )
            # 舊的分段仍在，只是沒有角色與峰值。
            (row,) = store.daily_report(days=0)
            self.assertEqual(row["character"], "")
            self.assertEqual(row["exp"], 1000)
            self.assertIsNone(row["peak_exp_per_hour"])
            self.assertIsNone(row["pct_gained"])


if __name__ == "__main__":
    unittest.main()
