"""速率估算與格式化測試。"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401

from mapleexp.core.stats import (
    ExpSeries,
    eta_seconds,
    format_duration,
    format_elapsed,
    format_exp,
    format_rate,
    format_window,
    linregress_slope,
)


class TestLinregress(unittest.TestCase):
    def test_perfect_line(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 10.0, 20.0, 30.0]
        self.assertAlmostEqual(linregress_slope(xs, ys), 10.0)

    def test_noise_is_averaged_out(self):
        xs = [float(i) for i in range(21)]
        ys = [10.0 * i + (5 if i % 2 else -5) for i in range(21)]
        self.assertAlmostEqual(linregress_slope(xs, ys), 10.0, places=6)

    def test_insufficient_data(self):
        self.assertIsNone(linregress_slope([], []))
        self.assertIsNone(linregress_slope([1.0], [1.0]))

    def test_zero_variance_in_x(self):
        self.assertIsNone(linregress_slope([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


class TestExpSeries(unittest.TestCase):
    def test_rate_from_steady_gain(self):
        series = ExpSeries([60, 300])
        for i in range(61):
            series.add(float(i), 100 * i)
        rate = series.rate(60, 60.0)
        self.assertAlmostEqual(rate.exp_per_hour, 360_000.0, places=3)
        self.assertEqual(rate.exp_gained, 6000)
        self.assertTrue(rate.valid)

    def test_short_window_reacts_faster_than_long_window(self):
        series = ExpSeries([60, 600])
        # 前 10 分鐘每秒 100
        for i in range(601):
            series.add(float(i), 100 * i)
        # 接下來 1 分鐘每秒 500
        base = 100 * 600
        for i in range(1, 61):
            series.add(600.0 + i, base + 500 * i)
        short = series.rate(60, 660.0).exp_per_hour
        long = series.rate(600, 660.0).exp_per_hour
        self.assertAlmostEqual(short, 1_800_000.0, places=2)
        self.assertLess(long, short)
        self.assertGreater(long, 360_000.0)

    def test_window_excludes_old_points(self):
        series = ExpSeries([60])
        for i in range(31):
            series.add(float(i), 0)          # 前 30 秒完全沒有經驗
        for i in range(1, 31):
            series.add(30.0 + i, 100 * i)    # 後 30 秒每秒 100
        rate = series.rate(30, 60.0).exp_per_hour
        self.assertAlmostEqual(rate, 360_000.0, places=2)

    def test_negative_slope_is_clamped_to_zero(self):
        series = ExpSeries([60])
        series.add(0.0, 1000)
        series.add(1.0, 900)
        self.assertEqual(series.rate(60, 1.0).exp_per_hour, 0.0)

    def test_empty_series(self):
        series = ExpSeries([60])
        rate = series.rate(60)
        self.assertFalse(rate.valid)
        self.assertEqual(rate.samples, 0)

    def test_memory_is_bounded(self):
        series = ExpSeries([60])
        for i in range(10_000):
            series.add(float(i), i)
        # 保留量應該接近視窗長度，不會無限成長
        self.assertLess(len(series), 300)

    def test_rates_covers_all_windows(self):
        series = ExpSeries([60, 300, 900])
        for i in range(101):
            series.add(float(i), 10 * i)
        rates = series.rates(100.0)
        self.assertEqual(sorted(rates), [60, 300, 900])


class TestEta(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(eta_seconds(3600, 3600.0), 3600.0)

    def test_already_complete(self):
        self.assertEqual(eta_seconds(0, 1000.0), 0.0)

    def test_no_rate(self):
        self.assertIsNone(eta_seconds(1000, None))
        self.assertIsNone(eta_seconds(1000, 0.0))

    def test_no_remaining_data(self):
        self.assertIsNone(eta_seconds(None, 1000.0))


class TestFormatting(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(format_duration(None), "--")
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(754), "12m34s")
        self.assertEqual(format_duration(5000), "1h23m")

    def test_elapsed_keeps_seconds_at_hour_scale(self):
        """活躍／閒置是實際累積的時間，到小時也要看得到秒。"""
        self.assertEqual(format_elapsed(45), "45s")
        self.assertEqual(format_elapsed(754), "12m34s")
        self.assertEqual(format_elapsed(5025), "1h23m45s")
        self.assertEqual(format_elapsed(3600), "1h00m00s")
        self.assertEqual(format_elapsed(None), "--")
        self.assertEqual(format_elapsed(-5), "--")

    def test_elapsed_and_duration_differ_only_above_an_hour(self):
        # 倒數估計值不需要秒（假精度），經過時間需要。
        self.assertEqual(format_duration(5025), "1h23m")
        self.assertEqual(format_elapsed(5025), "1h23m45s")
        for seconds in (0, 45, 754, 3599):
            self.assertEqual(format_duration(seconds), format_elapsed(seconds))

    def test_exp(self):
        self.assertEqual(format_exp(None), "--")
        self.assertEqual(format_exp(1234567), "1,234,567")

    def test_rate_uses_full_numbers(self):
        # 不用 k/M 縮寫：要比較兩個速率差多少時，縮寫得先在腦袋裡換算。
        self.assertEqual(format_rate(None), "--")
        self.assertEqual(format_rate(5000), "5,000/h")
        self.assertEqual(format_rate(45_000), "45,000/h")
        self.assertEqual(format_rate(2_500_000), "2,500,000/h")

    def test_window_label(self):
        self.assertEqual(format_window(300), "5 分鐘")
        self.assertEqual(format_window(900), "15 分鐘")
        self.assertEqual(format_window(3600), "1 小時")
        self.assertEqual(format_window(5400), "1.5 小時")
        self.assertEqual(format_window(30), "30 秒")
        self.assertEqual(format_window(None), "--")


if __name__ == "__main__":
    unittest.main()
