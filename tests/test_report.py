"""報表視窗的排序與篩選邏輯（不開視窗，只測純函式）。"""

from __future__ import annotations

import time
import unittest

import helpers  # noqa: F401

from mapleexp.ui.report import _parse_date, _passes, _row_from_record, _sort_key


def record(**overrides) -> dict:
    base = dict(
        session_id=1, map_id="m", map_name="樹林底層", character="甲", job="獵人",
        started_at=1_700_000_000.0, ended_at=None, exp=60_000,
        active_sec=600.0, pct_gained=14.0, peak_exp_per_hour=420_000.0, deaths=0, levelups=0,
    )
    base.update(overrides)
    return base


NO_FILTER = {"character": None, "job": None, "map": None}


class TestRows(unittest.TestCase):
    def test_rate_is_derived_and_formatted(self):
        row = _row_from_record(record())
        self.assertAlmostEqual(row["raw"]["rate"], 360_000.0)
        self.assertEqual(row["shown"]["rate"], "360,000/h")
        self.assertEqual(row["shown"]["pct"], "+14.00%")
        self.assertEqual(row["shown"]["active"], "10m00s")

    def test_missing_values_show_dashes(self):
        row = _row_from_record(record(pct_gained=None, peak_exp_per_hour=None, active_sec=0.0))
        self.assertEqual(row["shown"]["pct"], "--")
        self.assertEqual(row["shown"]["peak"], "--")
        self.assertIsNone(row["raw"]["rate"])

    def test_unnamed_map_shows_short_id(self):
        row = _row_from_record(record(map_name="", map_id="abcdef123456"))
        self.assertEqual(row["shown"]["map"], "(未命名 abcdef)")


class TestFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.row = _row_from_record(record())

    def test_no_filter_passes(self):
        self.assertTrue(_passes(self.row, NO_FILTER, (None, None)))

    def test_set_filter_uses_displayed_text(self):
        self.assertTrue(_passes(self.row, {**NO_FILTER, "character": {"甲", "乙"}}, (None, None)))
        self.assertFalse(_passes(self.row, {**NO_FILTER, "character": {"乙"}}, (None, None)))
        self.assertFalse(_passes(self.row, {**NO_FILTER, "map": set()}, (None, None)))

    def test_unknown_character_is_filtered_by_its_dash(self):
        row = _row_from_record(record(character=""))
        self.assertTrue(_passes(row, {**NO_FILTER, "character": {"-"}}, (None, None)))

    def test_date_range_is_start_inclusive_end_exclusive(self):
        started = self.row["raw"]["started"]
        self.assertTrue(_passes(self.row, NO_FILTER, (started, started + 1)))
        self.assertFalse(_passes(self.row, NO_FILTER, (started + 1, None)))
        self.assertFalse(_passes(self.row, NO_FILTER, (None, started)))
        self.assertTrue(_passes(self.row, NO_FILTER, (None, started + 1)))


class TestParseDate(unittest.TestCase):
    def test_accepts_common_spellings(self):
        expected = time.mktime(time.strptime("2026-10-07", "%Y-%m-%d"))
        for text in ("2026-10-07", "2026/10/7", "20261007", "2026.10.07", " 2026-10-07 "):
            self.assertEqual(_parse_date(text), expected, text)

    def test_empty_means_unbounded(self):
        self.assertIsNone(_parse_date(""))
        self.assertIsNone(_parse_date("   "))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            _parse_date("昨天")


class TestSort(unittest.TestCase):
    def test_none_sorts_last_in_ascending_order(self):
        with_peak = _row_from_record(record(peak_exp_per_hour=1.0))
        without = _row_from_record(record(peak_exp_per_hour=None))
        self.assertLess(_sort_key(with_peak, "peak", "num"), _sort_key(without, "peak", "num"))

    def test_text_sort_ignores_case(self):
        a = _row_from_record(record(character="abc"))
        b = _row_from_record(record(character="ABD"))
        self.assertLess(_sort_key(a, "character", "text"), _sort_key(b, "character", "text"))


if __name__ == "__main__":
    unittest.main()
