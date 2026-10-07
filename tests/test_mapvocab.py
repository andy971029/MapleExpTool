"""地圖名稱比對測試。

比對只有一個判斷：拿 OCR 讀到的整串文字，跟每張地圖的「地區＋地圖名」算編輯
距離，挑最接近的。沒有門檻 —— 一定會給答案，錯誤率由實際使用去量。
"""

from __future__ import annotations

import unittest

import helpers  # noqa: F401

from mapleexp.core import mapvocab
from mapleexp.gamedata import MapRecord, MapVocabulary

# 真實清單的縮影：同地區裡有互為前綴、只差一個字的名字，那是最容易混的情況。
VOCAB = MapVocabulary(
    (
        MapRecord("1", "迷霧森林", "奇幻村"),
        MapRecord("2", "迷霧森林", "奇幻村旅館"),
        MapRecord("3", "戰火之地", "沼澤地Ⅰ"),
        MapRecord("4", "戰火之地", "沼澤地Ⅱ"),
        MapRecord("5", "隱密之地", "樹林底層"),
        MapRecord("6", "隱藏地圖", "廢棄的妖精的圖書館"),
        MapRecord("7", "楓之島", "菇菇村"),
        MapRecord("8", "楓之島", "菇菇村"),  # 同名不同 id，真實資料裡很常見
    )
)


class TestEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(mapvocab.edit_distance("奇幻村", "奇幻村"), 0)

    def test_insertion(self):
        self.assertEqual(mapvocab.edit_distance("奇幻村", "奇幻村旅館"), 2)

    def test_substitution(self):
        self.assertEqual(mapvocab.edit_distance("沼澤地Ⅰ", "沼澤地Ⅱ"), 1)

    def test_empty(self):
        self.assertEqual(mapvocab.edit_distance("", "奇幻村"), 3)
        self.assertEqual(mapvocab.edit_distance("奇幻村", ""), 3)
        self.assertEqual(mapvocab.edit_distance("", ""), 0)

    def test_ceiling_stops_early_but_keeps_the_comparison_valid(self):
        """提早放棄只需要保證「超過門檻」這件事還是對的。"""
        self.assertGreater(mapvocab.edit_distance("甲乙丙丁", "戊己庚辛", ceiling=1), 1)

    def test_ceiling_does_not_change_an_answer_below_it(self):
        self.assertEqual(mapvocab.edit_distance("奇幻村", "奇幻村旅館", ceiling=5), 2)


class TestSimilarity(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(mapvocab.similarity("奇幻村", "奇幻村"), 1.0)

    def test_extra_characters_cost(self):
        """字數差要扣分 —— 這正是舊做法分不開前綴名稱的原因。"""
        self.assertAlmostEqual(mapvocab.similarity("奇幻村", "奇幻村旅館"), 1 - 2 / 5)

    def test_both_empty(self):
        self.assertEqual(mapvocab.similarity("", ""), 1.0)

    def test_nothing_in_common(self):
        self.assertEqual(mapvocab.similarity("甲乙", "丙丁"), 0.0)


class TestClean(unittest.TestCase):
    def test_strips_ocr_noise(self):
        self.assertEqual(mapvocab.clean("樹林|底層,"), "樹林底層")

    def test_expands_roman_numerals(self):
        """客戶端自己就混用兩種寫法：「森林迷宮I」是 ASCII，「猴子森林Ⅱ」是
        Unicode 羅馬數字。不展開的話，OCR 讀到哪一種就配不上另一種。"""
        self.assertEqual(mapvocab.clean("沼澤地Ⅱ"), "沼澤地II")
        self.assertEqual(mapvocab.clean("沼澤地II"), "沼澤地II")

    def test_roman_numeral_lookalikes_are_folded(self):
        """實測：Ⅱ 被讀成 LL、Ⅲ 被讀成 川。"""
        self.assertEqual(mapvocab.clean("沼澤地LL"), "沼澤地II")
        self.assertEqual(mapvocab.clean("沼澤地ll"), "沼澤地II")
        self.assertEqual(mapvocab.clean("沼澤地川"), "沼澤地III")

    def test_uppercases_for_comparison(self):
        """只影響比對；回傳的仍然是客戶端原本的寫法。"""
        self.assertEqual(mapvocab.clean("Zone 51"), "ZONE51")

    def test_full_width_characters_are_folded(self):
        self.assertEqual(mapvocab.clean("ＶＩＰ"), "VIP")

    def test_empty(self):
        self.assertEqual(mapvocab.clean(""), "")
        self.assertEqual(mapvocab.clean(None), "")


class TestIdentify(unittest.TestCase):
    def test_perfect_read(self):
        guess = mapvocab.identify("迷霧森林\n奇幻村", VOCAB)
        self.assertEqual(guess.street, "迷霧森林")
        self.assertEqual(guess.name, "奇幻村")
        self.assertEqual(guess.score, 1.0)
        self.assertEqual(guess.full_name, "迷霧森林 奇幻村")

    def test_prefix_of_a_longer_sibling_is_resolved(self):
        """「奇幻村」也是「奇幻村旅館」的前綴 —— 字數差讓它們分得開。

        舊做法用「共同子序列佔讀到字數的比例」，兩者同分、領先 0，只能放棄。
        """
        guess = mapvocab.identify("迷霧森林\n奇幻村", VOCAB)
        self.assertEqual(guess.name, "奇幻村")
        self.assertEqual(guess.runner_up, "迷霧森林 奇幻村旅館")
        self.assertGreater(guess.lead, 0.0)

    def test_merged_lines_match_the_same(self):
        """OCR 有時把兩行併成一條回傳；接成一串比就不必管這件事。"""
        self.assertEqual(
            mapvocab.identify("迷霧森林奇幻村", VOCAB).name,
            mapvocab.identify("迷霧森林\n奇幻村", VOCAB).name,
        )

    def test_missing_street_line_still_matches(self):
        guess = mapvocab.identify("奇幻村", VOCAB)
        self.assertEqual(guess.name, "奇幻村")
        self.assertLess(guess.score, 1.0)

    def test_partial_and_misread_characters(self):
        """實機讀到的樣子：缺字加讀錯字。"""
        self.assertEqual(mapvocab.identify("火之圯沼澤", VOCAB).street, "戰火之地")

    def test_long_name_split_into_extra_lines(self):
        guess = mapvocab.identify("隱藏地圖\n廢棄的妖精\n的圖書館", VOCAB)
        self.assertEqual(guess.name, "廢棄的妖精的圖書館")

    def test_always_answers_even_for_a_poor_read(self):
        """刻意不設門檻：先量實際錯誤率，再決定要不要加關卡。"""
        guess = mapvocab.identify("的的同", VOCAB)
        self.assertTrue(guess.name)
        self.assertLess(guess.score, 0.5)

    def test_nothing_read_is_not_a_guess(self):
        """沒有輸入就沒有東西可比 —— 這不是「選不出來」。"""
        for text in ("", "   ", "|||", "\n\n"):
            guess = mapvocab.identify(text, VOCAB)
            self.assertEqual(guess.name, "", text)
            self.assertEqual(guess.full_name, "", text)

    def test_empty_vocabulary_is_not_an_error(self):
        guess = mapvocab.identify("迷霧森林\n奇幻村", MapVocabulary())
        self.assertEqual(guess.name, "")

    def test_read_is_reported_for_comparison(self):
        """實測時要能看到 OCR 到底讀到什麼，才知道錯在哪一段。"""
        self.assertEqual(mapvocab.identify("迷霧森林|\n奇幻村", VOCAB).read, "迷霧森林奇幻村")

    def test_duplicate_records_do_not_become_their_own_runner_up(self):
        """「菇菇村」在清單裡出現兩次，不該讓領先差距變成 0。"""
        guess = mapvocab.identify("楓之島菇菇村", VOCAB)
        self.assertEqual(guess.name, "菇菇村")
        self.assertNotEqual(guess.runner_up, "楓之島 菇菇村")
        self.assertGreater(guess.lead, 0.0)

    def test_roman_numerals_match_across_both_spellings(self):
        """實機踩到的：OCR 讀到「Ⅲ」而客戶端寫的是「III」。

        正規化之前這兩者永遠配不上，只能在 I/II/III/IV/V 裡亂挑（領先 0）。
        """
        numbered = MapVocabulary(
            (
                MapRecord("1", "迷霧森林", "森林迷宮I"),
                MapRecord("2", "迷霧森林", "森林迷宮II"),
                MapRecord("3", "迷霧森林", "森林迷宮III"),
            )
        )
        guess = mapvocab.identify("迷霧森林森林迷宮Ⅲ", numbered)
        self.assertEqual(guess.name, "森林迷宮III")
        self.assertGreater(guess.lead, 0.0)

    def test_misread_roman_numerals_pick_the_right_sibling(self):
        """實機紀錄：``沼澤地LL`` 跟 Ⅰ、Ⅱ 同分被硬挑成 Ⅰ，``沼澤地川`` 也配成 Ⅰ。"""
        vocab = MapVocabulary(
            (
                MapRecord("1", "戰火之地", "沼澤地Ⅰ"),
                MapRecord("2", "戰火之地", "沼澤地Ⅱ"),
                MapRecord("3", "戰火之地", "沼澤地Ⅲ"),
            )
        )
        self.assertEqual(mapvocab.identify("戰火之地沼澤地LL", vocab).name, "沼澤地Ⅱ")
        self.assertEqual(mapvocab.identify("戰火之地沼澤地川", vocab).name, "沼澤地Ⅲ")
        self.assertEqual(mapvocab.identify("戰火之地沼澤地l", vocab).name, "沼澤地Ⅰ")

    def test_a_genuine_tie_shows_a_zero_lead(self):
        """真的分不開時還是會給答案（刻意的），但 lead=0 看得出來這次是硬猜的。

        要構成真正的平手，兩個候選得**一樣長**且只差那一個讀錯的字；長度不同的
        話，編輯距離本身就會把長的那個扣得比較多（這正是前綴那題能解開的原因）。
        """
        zones = MapVocabulary(
            (
                MapRecord("1", "墮落廣場", "VIP zone A區域"),
                MapRecord("2", "墮落廣場", "VIP zone C區域"),
            )
        )
        guess = mapvocab.identify("墮落廣場VIP zone 口區域", zones)
        self.assertIn(guess.name, ("VIP zone A區域", "VIP zone C區域"))
        self.assertEqual(guess.lead, 0.0)

    def test_matches_the_brute_force_answer(self):
        """提早放棄不能改變答案，也不能弄壞第二名。"""
        for text in ("迷霧森林奇幻村", "奇幻村", "的的同", "樹林", "沼澤"):
            read = mapvocab.clean(text)
            ranked = sorted(
                {
                    mapvocab.clean(r.street + r.name): f"{r.street} {r.name}"
                    for r in VOCAB.records
                }.items(),
                key=lambda kv: -mapvocab.similarity(read, kv[0]),
            )
            guess = mapvocab.identify(text, VOCAB)
            top = mapvocab.similarity(read, ranked[0][0])
            second = mapvocab.similarity(read, ranked[1][0])
            self.assertAlmostEqual(guess.score, top, msg=text)
            self.assertAlmostEqual(guess.lead, top - second, msg=text)


if __name__ == "__main__":
    unittest.main()
