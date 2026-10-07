"""追蹤狀態機測試。

這些測試完全不需要遊戲或截圖 —— 直接餵合成的取樣序列進去。
"""

from __future__ import annotations

import unittest

from helpers import miss, reading

from mapleexp.config import TrackerConfig
from mapleexp.core.tracker import ACTIVE, IDLE, PAUSED, Tracker

NEED = 10_000


def make_tracker(**overrides) -> Tracker:
    cfg = TrackerConfig(
        sample_interval=1.0,
        miss_to_pause=3,
        max_bridge_sec=30.0,
        outlier_level_frac=0.25,
        confirm_surprises=True,
        rate_windows=[60, 300],
        eta_window=60,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return Tracker(cfg)


def feed_all(tracker: Tracker, readings) -> list:
    events = []
    for r in readings:
        events.extend(tracker.feed(r))
    return events


class TestDiscontinuity(unittest.TestCase):
    """換角色：兩隻角色的經驗值之間沒有可比較的關係。"""

    def test_next_reading_becomes_a_fresh_baseline(self):
        t = make_tracker()
        feed_all(t, [reading(0, 1000, need=NEED), reading(1, 1100, need=NEED)])
        t.mark_discontinuity()
        events = feed_all(
            t,
            [
                reading(2, 50, need=5_000),      # 另一隻角色，經驗低得多
                reading(3, 150, need=5_000),
            ],
        )
        snap = t.snapshot()
        self.assertEqual([e.kind for e in events], ["rebase"])
        self.assertEqual(snap.deaths, 0)
        self.assertEqual(snap.exp_lost, 0)
        self.assertEqual(snap.cum_gross, 200)        # 100（第一隻）+ 100（第二隻）
        self.assertEqual(snap.exp_abs, 150)

    def test_discontinuity_drops_a_pending_suspect(self):
        """換角色時扣著待確認的取樣已經沒有意義，不能拿新角色的取樣去「確認」它。"""
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED),
                reading(1, 900, need=NEED),      # 疑似死亡，等確認
            ],
        )
        t.mark_discontinuity()
        feed_all(t, [reading(2, 50, need=5_000), reading(3, 150, need=5_000)])
        snap = t.snapshot()
        self.assertEqual(snap.deaths, 0)
        self.assertEqual(snap.cum_gross, 100)

    def test_without_discontinuity_the_switch_looks_like_a_death(self):
        """對照組：不標記的話，追蹤器只能把它當成經驗暴跌。"""
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED),
                reading(1, 1100, need=NEED),
                reading(2, 50, need=5_000),
                reading(3, 150, need=5_000),
            ],
        )
        self.assertEqual(t.snapshot().deaths, 1)


class TestSteadyGain(unittest.TestCase):
    def test_accumulates_and_computes_rate(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED, level=30),
                reading(1, 1100, need=NEED, level=30),
                reading(2, 1200, need=NEED, level=30),
                reading(3, 1300, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.state, ACTIVE)
        self.assertEqual(snap.cum_gross, 300)
        self.assertEqual(snap.cum_net, 300)
        self.assertAlmostEqual(snap.active_sec, 3.0)
        self.assertEqual(snap.deaths, 0)
        self.assertEqual(snap.levelups, 0)
        # 100 exp/sec -> 360,000 exp/hr
        self.assertAlmostEqual(snap.rates[60].exp_per_hour, 360_000.0, places=3)

    def test_need_and_remaining_are_derived(self):
        t = make_tracker()
        feed_all(t, [reading(0, 2500, need=NEED, level=30)])
        snap = t.snapshot()
        self.assertEqual(snap.need, NEED)
        self.assertEqual(snap.remaining, NEED - 2500)

    def test_eta_uses_rate_and_remaining(self):
        t = make_tracker()
        feed_all(
            t,
            [reading(i, 1000 + 100 * i, need=NEED, level=30) for i in range(4)],
        )
        snap = t.snapshot()
        # 剩 10000-1300 = 8700，速率 100/秒 -> 87 秒
        self.assertIsNotNone(snap.eta_sec)
        self.assertAlmostEqual(snap.eta_sec, 87.0, places=2)


class TestLevelUp(unittest.TestCase):
    def test_level_up_bridges_remaining_exp(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 9700, need=NEED, level=30),
                reading(1, 9800, need=NEED, level=30),
                reading(2, 150, need=20_000, level=31),   # 升級（待確認）
                reading(3, 350, need=20_000, level=31),   # 確認
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.levelups, 1)
        self.assertEqual(snap.deaths, 0)
        # 100 (9700->9800) + 350 (9800->滿 10000 再到 150) + 200 (150->350)
        self.assertEqual(snap.cum_gross, 650)
        self.assertEqual(snap.level, 31)

    def test_level_indicator_catching_up_late_is_not_a_second_levelup(self):
        """升級已經從經驗歸零算過了；等級數字晚幾格才跟上（新字形要先學會）。

        那幾格的取樣是「等級 +1 但經驗沒歸零」，不能再算一次升級 —— 否則會把一整級
        的經驗重複加進去。
        """
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 9700, need=NEED, level=30),
                reading(1, 9800, need=NEED, level=30),
                reading(2, 150, need=20_000, level=30),    # 歸零，但等級還讀不到新字形
                reading(3, 350, need=20_000, level=30),    # 確認
                reading(4, 450, need=20_000, level=31),    # 等級數字跟上了
                reading(5, 550, need=20_000, level=31),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.levelups, 1)
        self.assertEqual(snap.glitches, 0)
        # 100 + (200 + 150) + 200 + 100 + 100
        self.assertEqual(snap.cum_gross, 850)
        self.assertEqual(snap.level, 31)

    def test_level_up_detected_without_level(self):
        """取樣裡沒有等級時，靠「掉掉幾乎全部」的特徵判定升級。"""
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 9700, need=NEED),
                reading(1, 9800, need=NEED),
                reading(2, 150, need=20_000),
                reading(3, 350, need=20_000),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.levelups, 1)
        self.assertEqual(snap.deaths, 0)
        self.assertEqual(snap.cum_gross, 650)

    def test_multi_level_jump_is_marked_uncertain(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 9800, need=NEED, level=30),
                reading(1, 150, need=40_000, level=33),
                reading(2, 350, need=40_000, level=33),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.levelups, 3)
        self.assertGreaterEqual(snap.uncertain, 1)


class TestDeath(unittest.TestCase):
    def test_exp_loss_recorded_and_net_reduced(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 5000, need=NEED, level=30),
                reading(1, 5200, need=NEED, level=30),
                reading(2, 4800, need=NEED, level=30),  # 死亡（待確認）
                reading(3, 4900, need=NEED, level=30),  # 確認
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.deaths, 1)
        self.assertEqual(snap.exp_lost, 400)
        self.assertEqual(snap.cum_gross, 300)   # 200 + 100
        self.assertEqual(snap.cum_net, -100)    # 200 - 400 + 100
        self.assertEqual(snap.levelups, 0)

    def test_small_drop_is_not_mistaken_for_level_up(self):
        """等級未知時，小幅下降必須判定為死亡而不是升級。"""
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 6000, need=NEED),
                reading(1, 5000, need=NEED),
                reading(2, 5100, need=NEED),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.deaths, 1)
        self.assertEqual(snap.levelups, 0)
        self.assertEqual(snap.exp_lost, 1000)


class TestGlitchFiltering(unittest.TestCase):
    def test_single_frame_spike_is_discarded(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 5000, need=NEED, level=30),
                reading(1, 5100, need=NEED, level=30),
                reading(2, 99_999, pct=99.99, level=30),  # 誤判
                reading(3, 5200, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.glitches, 1)
        self.assertEqual(snap.cum_gross, 200)  # 100 + 100，誤判那格被丟掉
        self.assertEqual(snap.deaths, 0)

    def test_single_frame_dip_is_not_counted_as_death(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 5000, need=NEED, level=30),
                reading(1, 5100, need=NEED, level=30),
                reading(2, 100, pct=1.0, level=30),  # 誤判成很小的數
                reading(3, 5200, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.deaths, 0)
        self.assertEqual(snap.levelups, 0)
        self.assertEqual(snap.glitches, 1)
        self.assertEqual(snap.cum_gross, 200)

    def test_confirmation_can_be_disabled(self):
        t = make_tracker(confirm_surprises=False)
        feed_all(
            t,
            [
                reading(0, 5000, need=NEED, level=30),
                reading(1, 4800, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.deaths, 1)
        self.assertEqual(snap.glitches, 0)

    def test_outlier_rejected_by_level_fraction(self):
        """增量超過本級所需經驗的設定比例就視為誤判。"""
        t = make_tracker(outlier_level_frac=0.10)
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED, level=30),
                reading(1, 3000, need=NEED, level=30),  # +2000 > 10000*0.10
                reading(2, 1100, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(snap.glitches, 1)
        self.assertEqual(snap.cum_gross, 100)


class TestPauseAndGaps(unittest.TestCase):
    def test_consecutive_misses_pause_the_clock(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED, level=30),
                reading(1, 1100, need=NEED, level=30),
                miss(2),
                miss(3),
                miss(4),
            ],
        )
        self.assertEqual(t.state, PAUSED)
        snap = t.snapshot()
        # 失敗的取樣不會推進活躍時間
        self.assertAlmostEqual(snap.active_sec, 1.0)
        self.assertEqual(snap.misses, 3)

    def test_short_interruption_is_bridged(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED, level=30),
                reading(1, 1100, need=NEED, level=30),
                miss(2),
                miss(3),
                miss(4),
                reading(5, 1500, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        self.assertEqual(t.state, ACTIVE)
        # 中斷 4 秒 <= max_bridge_sec，經驗與時間都照算
        self.assertEqual(snap.cum_gross, 500)
        self.assertAlmostEqual(snap.active_sec, 5.0)
        self.assertAlmostEqual(snap.skipped_sec, 0.0)

    def test_long_interruption_excludes_both_exp_and_time(self):
        t = make_tracker()
        feed_all(
            t,
            [
                reading(0, 1000, need=NEED, level=30),
                reading(1, 1100, need=NEED, level=30),
                reading(200, 5000, need=NEED, level=30),  # 中斷 199 秒
                reading(201, 5100, need=NEED, level=30),
                reading(202, 5200, need=NEED, level=30),
            ],
        )
        snap = t.snapshot()
        # 第一段 100 + 恢復後的 100 + 100；中斷那段 3900 不列入
        self.assertEqual(snap.cum_gross, 300)
        self.assertAlmostEqual(snap.active_sec, 3.0)
        self.assertAlmostEqual(snap.skipped_sec, 199.0)

    def test_rate_is_not_diluted_by_afk(self):
        """掛機後速率不應被稀釋 —— 這是整個設計最重要的性質。"""
        t = make_tracker()
        # 打 10 秒，每秒 100 經驗
        feed_all(t, [reading(i, 1000 + 100 * i, need=NEED, level=30) for i in range(11)])
        before = t.snapshot().rates[60].exp_per_hour
        # 掛機 5 分鐘（辨識失敗），再回來
        feed_all(t, [miss(11 + i) for i in range(300)])
        feed_all(t, [reading(320 + i, 2000 + 100 * i, need=NEED, level=30) for i in range(5)])
        after = t.snapshot().rates[60].exp_per_hour
        self.assertAlmostEqual(before, 360_000.0, places=3)
        self.assertAlmostEqual(after, 360_000.0, places=3)


class TestIdlePause(unittest.TestCase):
    """經驗值停滯時要停止累計活躍時間。

    走路、補血、聊天、掛機都會讓經驗停住；把那段時間算進分母，
    EXP/hr 就變成一個沒有意義的平均值。
    """

    def test_stall_stops_the_clock_after_threshold(self):
        t = make_tracker(idle_pause_sec=10.0)
        # 先正常打 5 秒
        feed_all(t, [reading(i, 1000 + 100 * i, need=NEED, level=30) for i in range(6)])
        # 接著 30 秒經驗完全不動（畫面還是讀得到）
        feed_all(t, [reading(6 + i, 1500, need=NEED, level=30) for i in range(30)])
        snap = t.snapshot()
        self.assertEqual(t.state, IDLE)
        # 5 秒正常 + 停滯後最多再算 10 秒
        self.assertAlmostEqual(snap.active_sec, 15.0, places=6)
        self.assertAlmostEqual(snap.idle_sec, 20.0, places=6)

    def test_starts_idle_until_the_first_gain(self):
        """開程式不等於開始練：拿到第一筆經驗之前全部是閒置，一秒都不算。"""
        t = make_tracker(idle_pause_sec=10.0)
        feed_all(t, [reading(0, 1000, need=NEED, level=30)])
        self.assertEqual(t.state, IDLE)
        feed_all(t, [reading(i, 1000, need=NEED, level=30) for i in range(1, 21)])
        snap = t.snapshot()
        self.assertEqual(t.state, IDLE)
        self.assertAlmostEqual(snap.active_sec, 0.0, places=6)
        self.assertAlmostEqual(snap.idle_sec, 20.0, places=6)

    def test_disabled_idle_detection_starts_active(self):
        t = make_tracker(idle_pause_sec=0.0)
        feed_all(t, [reading(0, 1000, need=NEED, level=30)])
        self.assertEqual(t.state, ACTIVE)

    def test_misses_while_idle_become_unreadable(self):
        """閒置中也會讀不到畫面（切頻、選角色），狀態要能變成「無法讀取」。"""
        t = make_tracker(idle_pause_sec=10.0, miss_to_pause=3)
        feed_all(t, [reading(0, 1000, need=NEED, level=30)])
        self.assertEqual(t.state, IDLE)
        events = feed_all(t, [miss(1), miss(2), miss(3)])
        self.assertEqual(t.state, PAUSED)
        self.assertEqual([e.kind for e in events], ["pause"])

    def test_resuming_gain_counts_its_whole_interval(self):
        """結束停滯的那一筆要整段算進去，否則分母趨近零、速率會噴掉。"""
        t = make_tracker(idle_pause_sec=10.0)
        feed_all(t, [reading(0, 1000, need=NEED, level=30)])
        feed_all(t, [reading(i, 1000, need=NEED, level=30) for i in range(1, 60)])
        self.assertEqual(t.state, IDLE)
        feed_all(t, [reading(60, 1600, need=NEED, level=30)])
        snap = t.snapshot()
        self.assertEqual(t.state, ACTIVE)
        # 開場到第一隻怪之前全是閒置；只有恢復那一筆的 1 秒算活躍
        self.assertAlmostEqual(snap.active_sec, 1.0, places=6)
        self.assertAlmostEqual(snap.idle_sec, 59.0, places=6)
        self.assertEqual(snap.cum_gross, 600)

    def test_rate_freezes_once_idle_instead_of_decaying(self):
        """進入閒置後速率要「凍結」，不是繼續往下掉。

        這就是「暫停計算」的意思：你去補血、聊天、掛機，回來看到的應該還是
        剛剛打怪時的效率，而不是被發呆時間稀釋過的數字。
        """
        t = make_tracker(idle_pause_sec=10.0, rate_windows=[600], eta_window=600)
        feed_all(t, [reading(i, 1000 + 100 * i, need=NEED, level=30) for i in range(11)])
        self.assertAlmostEqual(t.snapshot().rates[600].exp_per_hour, 360_000.0, places=3)

        # 發呆 30 秒：門檻內的 10 秒仍計入，所以速率會降一些
        feed_all(t, [reading(11 + i, 2000, need=NEED, level=30) for i in range(30)])
        after_30 = t.snapshot().rates[600].exp_per_hour
        self.assertLess(after_30, 360_000.0)

        # 再發呆 2 分鐘：速率必須完全不變
        feed_all(t, [reading(41 + i, 2000, need=NEED, level=30) for i in range(120)])
        after_150 = t.snapshot().rates[600].exp_per_hour
        self.assertAlmostEqual(after_30, after_150, places=6)
        self.assertAlmostEqual(t.snapshot().active_sec, 20.0, places=6)

    def test_can_be_disabled(self):
        t = make_tracker(idle_pause_sec=0.0)
        feed_all(t, [reading(0, 1000, need=NEED, level=30)])
        feed_all(t, [reading(i, 1000, need=NEED, level=30) for i in range(1, 40)])
        snap = t.snapshot()
        self.assertEqual(t.state, ACTIVE)
        self.assertAlmostEqual(snap.active_sec, 39.0, places=6)
        self.assertAlmostEqual(snap.idle_sec, 0.0)

    def test_slow_kills_within_threshold_stay_active(self):
        """每 8 秒一隻怪、門檻 10 秒 -> 不該被判成閒置。"""
        t = make_tracker(idle_pause_sec=10.0)
        readings = []
        for kill in range(5):
            base = kill * 8
            for i in range(8):
                exp = 1000 + kill * 500 + (500 if i == 7 else 0)
                readings.append(reading(base + i, exp, need=NEED, level=30))
        feed_all(t, readings)
        self.assertEqual(t.state, ACTIVE)
        # 唯一的閒置是開場到第一隻怪打死之前的 6 秒（第 7 秒那一格算活躍）。
        self.assertAlmostEqual(t.snapshot().idle_sec, 6.0, places=6)
        self.assertAlmostEqual(t.snapshot().active_sec, 33.0, places=6)


class TestReset(unittest.TestCase):
    def test_reset_clears_everything(self):
        t = make_tracker()
        feed_all(t, [reading(i, 1000 + 100 * i, need=NEED, level=30) for i in range(4)])
        t.reset()
        snap = t.snapshot()
        self.assertEqual(snap.cum_gross, 0)
        self.assertEqual(snap.cum_net, 0)
        self.assertEqual(snap.active_sec, 0.0)
        self.assertIsNone(snap.level)


if __name__ == "__main__":
    unittest.main()
