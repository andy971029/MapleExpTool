"""狀態列字串解析與所需經驗推導的測試。"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401  (插入 src 到 sys.path)

from mapleexp.core.reading import StatusReading, parse_exp_text


class TestParseExpText(unittest.TestCase):
    def test_bracket_layout(self):
        self.assertEqual(parse_exp_text("12345[30.25%]"), (12345, 30.25))

    def test_bracket_layout_with_spaces_and_label(self):
        self.assertEqual(parse_exp_text("EXP 12345 [30.25%]"), (12345, 30.25))

    def test_thousands_separator_is_ignored(self):
        self.assertEqual(parse_exp_text("1,234,567[8.90%]"), (1234567, 8.9))

    def test_bracket_with_integer_percent(self):
        self.assertEqual(parse_exp_text("500[7%]"), (500, 7.0))

    def test_missing_decimal_point_is_recovered(self):
        """小數點只有 2x2 像素，很容易讀不到。

        這個遊戲的百分比固定兩位小數，所以括號內超過 100 就代表小數點漏了，
        把最後兩位還原成小數即可。少了這條，小數點一讀不到百分比就整個報廢，
        連帶推導不出升級所需經驗。
        """
        self.assertEqual(parse_exp_text("117757[2750%]"), (117757, 27.5))
        self.assertEqual(parse_exp_text("112141[2618%]"), (112141, 26.18))

    def test_integer_percentage_is_not_misread_as_lost_decimal(self):
        # 100 以下視為原本就是整數百分比，不做還原。
        self.assertEqual(parse_exp_text("500[7%]"), (500, 7.0))
        self.assertEqual(parse_exp_text("500[100%]"), (500, 100.0))

    def test_no_bracket_but_decimal_present(self):
        self.assertEqual(parse_exp_text("12345 30.25"), (12345, 30.25))

    def test_absolute_only(self):
        self.assertEqual(parse_exp_text("98765"), (98765, None))

    def test_percent_only(self):
        exp, pct = parse_exp_text("30.25")
        self.assertIsNone(exp)
        self.assertEqual(pct, 30.25)

    def test_empty_and_garbage(self):
        self.assertEqual(parse_exp_text(""), (None, None))
        self.assertEqual(parse_exp_text("---"), (None, None))

    def test_percent_out_of_range_is_rejected(self):
        exp, pct = parse_exp_text("12345[300.25%]")
        self.assertEqual(exp, 12345)
        self.assertIsNone(pct)

    def test_longest_digit_run_wins(self):
        # 括號外若混進別的短數字，取最長的那串。
        self.assertEqual(parse_exp_text("7 1234567[12.00%]"), (1234567, 12.0))


class TestDerivedNeed(unittest.TestCase):
    def test_derives_level_requirement_from_abs_and_pct(self):
        # 實際所需經驗 10000，目前 3025 -> 顯示 30.25%
        r = StatusReading(mono=0, wall=0, ok=True, exp_abs=3025, exp_pct=30.25)
        self.assertEqual(r.derived_need, 10000)

    def test_bounds_bracket_the_true_value(self):
        true_need = 1_234_567
        exp = 456_789
        pct = round(100.0 * exp / true_need, 2)
        r = StatusReading(mono=0, wall=0, ok=True, exp_abs=exp, exp_pct=pct)
        bounds = r.need_bounds
        self.assertIsNotNone(bounds)
        lo, hi = bounds
        self.assertLessEqual(lo, true_need)
        self.assertGreaterEqual(hi, true_need)

    def test_precision_improves_with_higher_percentage(self):
        need = 1_000_000
        widths = []
        for pct_target in (1.0, 10.0, 90.0):
            exp = int(need * pct_target / 100)
            r = StatusReading(
                mono=0, wall=0, ok=True, exp_abs=exp, exp_pct=round(100.0 * exp / need, 2)
            )
            lo, hi = r.need_bounds
            widths.append((hi - lo) / lo)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_no_derivation_when_percentage_too_small(self):
        r = StatusReading(mono=0, wall=0, ok=True, exp_abs=10, exp_pct=0.01)
        self.assertIsNone(r.derived_need)

    def test_no_derivation_without_percentage(self):
        r = StatusReading(mono=0, wall=0, ok=True, exp_abs=10_000, exp_pct=None)
        self.assertIsNone(r.derived_need)


if __name__ == "__main__":
    unittest.main()
