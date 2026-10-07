"""等級經驗表（區間交集推導）測試。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from mapleexp.core.exptable import ExpTable
from mapleexp.core.reading import StatusReading


def bounds_for(exp: int, need: int) -> tuple[float, float]:
    pct = round(100.0 * exp / need, 2)
    return StatusReading(mono=0, wall=0, ok=True, exp_abs=exp, exp_pct=pct).need_bounds


class TestIntervalIntersection(unittest.TestCase):
    def test_single_observation_brackets_truth(self):
        table = ExpTable()
        need = 1_234_567
        table.observe_bounds(80, bounds_for(400_000, need))
        self.assertLessEqual(abs(table.need(80) - need) / need, 0.02)

    def test_repeated_observations_converge(self):
        """多次觀測的區間交集必須單調收斂，而且永遠包住真值。"""
        table = ExpTable()
        need = 2_000_000
        widths = []
        for exp in (100_000, 350_000, 700_000, 1_500_000, 1_900_000):
            table.observe_bounds(90, bounds_for(exp, need))
            entry = table._needs[90]
            self.assertLessEqual(entry.lo, need)
            self.assertGreaterEqual(entry.hi, need)
            widths.append(entry.rel_width)
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertTrue(table._needs[90].reliable)
        self.assertEqual(table.confidence(90), "reliable")

    def test_estimate_is_accurate_after_convergence(self):
        table = ExpTable()
        need = 777_777
        for exp in (300_000, 500_000, 700_000):
            table.observe_bounds(50, bounds_for(exp, need))
        self.assertLessEqual(abs(table.need(50) - need) / need, 0.001)

    def test_conflicting_observation_is_ignored_once_established(self):
        table = ExpTable()
        need = 1_000_000
        for exp in (200_000, 500_000, 800_000):
            table.observe_bounds(60, bounds_for(exp, need))
        good = table.need(60)
        # 一筆明顯矛盾的觀測（像是辨識錯誤）不該推翻已經收斂的估計
        table.observe_bounds(60, (50.0, 60.0))
        self.assertEqual(table.need(60), good)

    def test_early_conflict_resets_estimate(self):
        table = ExpTable()
        table.observe_bounds(10, (100.0, 200.0))
        table.observe_bounds(10, (5000.0, 6000.0))
        self.assertGreater(table.need(10), 1000)

    def test_level_none_is_ignored(self):
        table = ExpTable()
        self.assertFalse(table.observe_bounds(None, (1.0, 2.0)))
        self.assertEqual(len(table), 0)

    def test_rough_estimate_is_not_reliable(self):
        table = ExpTable()
        # 1% 的觀測區間很寬
        table.observe_bounds(120, bounds_for(10_000, 1_000_000))
        self.assertEqual(table.confidence(120), "rough")
        self.assertIsNone(table.reliable_need(120))
        self.assertIsNotNone(table.need(120))


class TestManualValues(unittest.TestCase):
    def test_manual_wins_over_observations(self):
        table = ExpTable()
        table.set_manual(30, 98_280)
        table.observe_bounds(30, (1.0, 2.0))
        self.assertEqual(table.need(30), 98_280)
        self.assertEqual(table.confidence(30), "manual")
        self.assertEqual(table.reliable_need(30), 98_280)


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp_table.json"
            table = ExpTable(path)
            need = 500_000
            for exp in (100_000, 300_000, 450_000):
                table.observe_bounds(70, bounds_for(exp, need))
            table.set_manual(71, 600_000)
            table.save()

            reloaded = ExpTable(path)
            self.assertEqual(reloaded.need(71), 600_000)
            self.assertEqual(reloaded.confidence(71), "manual")
            self.assertEqual(reloaded.need(70), table.need(70))
            self.assertEqual(reloaded.known_levels(), [70, 71])

    def test_shorthand_json_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp_table.json"
            path.write_text('{"needs": {"30": 98280, "31": 112320}}', encoding="utf-8")
            table = ExpTable(path)
            self.assertEqual(table.need(30), 98_280)
            self.assertEqual(table.confidence(31), "manual")

    def test_corrupt_file_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp_table.json"
            path.write_text("not json at all", encoding="utf-8")
            table = ExpTable(path)
            self.assertEqual(len(table), 0)


if __name__ == "__main__":
    unittest.main()
