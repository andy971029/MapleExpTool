"""職業名稱比對測試。"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401

from mapleexp.core import jobvocab


class TestIdentify(unittest.TestCase):
    def test_exact_match(self):
        guess = jobvocab.identify("獵人")
        self.assertEqual(guess.name, "獵人")
        self.assertEqual(guess.score, 1.0)
        self.assertTrue(guess.matched)

    def test_one_wrong_char_is_repaired(self):
        """資料庫裡的實際紀錄：``槍騎兵`` 被 OCR 讀成 ``搶騎兵``。"""
        guess = jobvocab.identify("搶騎兵")
        self.assertEqual(guess.name, "槍騎兵")
        self.assertTrue(guess.matched)

    def test_gunslinger_misread_is_not_mistaken_for_brawler(self):
        """使用者回報：``槍手`` 被判成 ``打手``。

        OCR 把 ``槍`` 讀成 ``搶`` 後，``搶手`` 跟 ``槍手``、``打手`` 都只差一個字（0.5 同分），
        舊版照清單順序挑到排前面的 ``打手``。形近字先換掉就是完全命中。
        """
        for read in ("搶手", "鎗手", "抢手"):
            guess = jobvocab.identify(read)
            self.assertEqual(guess.name, "槍手", read)
            self.assertEqual(guess.score, 1.0, read)

    def test_tie_between_two_jobs_is_not_guessed(self):
        """只讀到一個 ``手`` 字：打手與槍手同分，沒有依據挑哪個，不該猜。"""
        guess = jobvocab.identify("手")
        self.assertFalse(guess.matched)
        self.assertEqual(guess.name, "手")

    def test_punctuation_variant_matches(self):
        """狀態列顯示 ``(火、毒)``，客戶端資料寫 ``(火/毒)``；標點不該影響比對。"""
        self.assertEqual(jobvocab.identify("巫師(火/毒)").name, "巫師(火、毒)")
        self.assertEqual(jobvocab.identify("巫師火毒").name, "巫師(火、毒)")

    def test_first_job_archer_is_listed(self):
        """客戶端那串字漏了弓箭手一轉，是依實機補上的。"""
        self.assertEqual(jobvocab.identify("弓箭手").name, "弓箭手")
        self.assertEqual(jobvocab.identify("弩弓手").name, "弩弓手")

    def test_short_exact_name_beats_longer_superstring(self):
        self.assertEqual(jobvocab.identify("騎士").name, "騎士")
        self.assertEqual(jobvocab.identify("劍士").name, "劍士")

    def test_unrelated_text_keeps_ocr_result(self):
        """讀到角色名之類不像職業的東西，不要硬配，保留原文。"""
        guess = jobvocab.identify("卍弘法艾德卍")
        self.assertFalse(guess.matched)
        self.assertEqual(guess.name, "卍弘法艾德卍")

    def test_empty(self):
        self.assertEqual(jobvocab.identify("").name, "")
        self.assertEqual(jobvocab.identify("  ").name, "")

    def test_every_listed_job_round_trips(self):
        for job in jobvocab.JOB_NAMES:
            self.assertEqual(jobvocab.identify(job).name, job)


if __name__ == "__main__":
    unittest.main()
