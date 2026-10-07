"""把 OCR 讀到的地圖名，對到客戶端清單裡最接近的那一張。

只有一個判斷：**拿 OCR 讀到的整串文字，跟每一張地圖的「地區＋地圖名」比，挑最
接近的**。沒有門檻、沒有信心分數的關卡 —— 一定會給一個答案。這是刻意的：先用
實際使用去量真正的錯誤率，再決定需不需要加關卡。

為什麼用編輯距離
----------------

``相似度 = 1 - 編輯距離 / 兩者較長的那個長度``

編輯距離同時涵蓋了**字數**跟**文字**：少一個字算一次刪除、多一個字算一次插入、
讀錯一個字算一次替換。所以「字數 + 文字一起比」不需要兩套規則，一個距離就夠了。

這剛好解掉舊做法解不掉的例子：OCR 讀到「迷霧森林奇幻村」時，

====================  ==========  ==========
候選                  編輯距離    相似度
====================  ==========  ==========
迷霧森林奇幻村        0           1.00
迷霧森林奇幻村旅館    2           0.78
====================  ==========  ==========

舊做法用的是「最長共同子序列佔讀到字數的比例」，刻意**不罰多出來的字**（因為當時
OCR 只讀得到三到五成），於是這兩個候選同分，分不出來。前處理改成解混合之後 OCR
幾乎全對，那個假設就反過來害人了。

為什麼比的是「地區＋地圖名」整串
--------------------------------

小地圖面板上面那行是地區（``streetName``）、下面是地圖名（``mapName``）。接成一串
再比，可以不用管 OCR 這次有沒有把兩行分開回傳 —— 兩行、一行、少讀一行，都是同一
個比法。

真正用來區分地圖的仍然是那塊像素的指紋，跟文字無關；名字只是給人看的，改名不會
動到任何統計。所以這裡猜錯的代價是「顯示的名字不對，使用者點一下改掉」。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from ..gamedata import MapVocabulary


def edit_distance(a: str, b: str, ceiling: int | None = None) -> int:
    """Levenshtein 距離。

    *ceiling* 是提早放棄的門檻：比對一千多張地圖時，大部分候選差得很遠，算完整個
    表格是浪費。這一列的最小值已經超過 ceiling 就不可能更好了，直接回傳。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # 刪除
                    current[j - 1] + 1,       # 插入
                    previous[j - 1] + (char_a != char_b),  # 替換
                )
            )
        if ceiling is not None and min(current) > ceiling:
            return ceiling + 1
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """0~1，1 表示完全一樣。字數差跟讀錯字都會扣分。"""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1.0 - edit_distance(a, b) / longest


def clean(text: str) -> str:
    """把一串文字正規化成可以直接比對的形式。

    做三件事：

    1. **NFKC 正規化。** 客戶端自己的編號寫法就不一致 —— 「森林迷宮I」用的是
       ASCII 的 ``I``（U+0049），「猴子森林Ⅱ」用的是 Unicode 羅馬數字
       ``Ⅱ``（U+2161），全清單裡兩種各有幾十個。OCR 讀到哪一種是看它高興，
       不正規化的話「讀到 Ⅲ」永遠配不上「III」，只能在 I/II/III/IV/V 裡亂挑。
       NFKC 會把 ``Ⅲ`` 展開成 ``III``，順便處理全形英數。
    2. **轉大寫。** 只影響比對，回傳的仍然是客戶端原本的寫法。
    3. **丟掉標點。** OCR 在筆畫黏成一團時會吐出 ``|``、``,``、``·`` 這類框線碎片，
       那些不是名稱的一部分。
    4. **換掉羅馬數字的形近字。** 673 張地圖有 119 張以 Ⅰ／Ⅱ／Ⅲ 結尾，而 OCR 會把
       ``Ⅱ`` 讀成 ``LL``、``Ⅲ`` 讀成 ``川``（實測 ``戰火之地沼澤地LL`` 跟 Ⅰ、Ⅱ 同分，
       硬挑成 Ⅰ）。清單裡沒有任何名字含 ``L`` 或 ``川``，所以直接換成 ``I``／``III``
       不會誤傷別的名字。
    """
    normalised = unicodedata.normalize("NFKC", text or "").upper()
    normalised = normalised.translate(_ROMAN_LOOKALIKES)
    return "".join(char for char in normalised if char.isalnum())


# OCR 常把羅馬數字看成的字 -> 正確的羅馬數字（NFKC + 大寫之後套用）。
_ROMAN_LOOKALIKES = str.maketrans(
    {"L": "I", "丨": "I", "〡": "I", "川": "III", "〣": "III"}
)


@dataclass(frozen=True)
class MapGuess:
    """比對結果。``name`` 一定有值（除非 OCR 什麼都沒讀到）。"""

    street: str = ""
    name: str = ""
    score: float = 0.0
    """跟選中那張地圖的相似度，1.0 表示一字不差。"""

    runner_up: str = ""
    """第二接近的那張（「地區 地圖名」），拿來判斷這次有多模稜兩可。"""

    lead: float = 0.0
    """跟第二名的相似度差距。"""

    read: str = ""
    """OCR 實際讀到什麼（清理過），對照用。"""

    @property
    def full_name(self) -> str:
        if self.street and self.name:
            return f"{self.street} {self.name}"
        return self.name or self.street


def identify(text: str, vocab: MapVocabulary) -> MapGuess:
    """挑出清單裡跟 OCR 結果最接近的一張地圖。

    OCR 什麼都沒讀到時回傳空的結果 —— 那不是「選不出來」，是根本沒有東西可以比。
    其餘情況一定會給答案，由使用者實際使用去檢驗錯誤率。
    """
    read = clean("".join((text or "").splitlines()))
    if not read or not vocab:
        return MapGuess(read=read)

    best_score = -1.0
    best: tuple[str, str] | None = None
    second = -1.0
    second_text = ""
    seen: set[str] = set()

    for record in vocab.records:
        combined = clean(record.street + record.name)
        if combined in seen:
            continue
        seen.add(combined)
        longest = max(len(read), len(combined))
        if longest == 0:
            continue
        # 連第二名都贏不了的候選，距離算到一半就可以停 —— 但門檻要用**第二名**
        # 而不是第一名，否則提早放棄回傳的高估值會把亞軍也污染掉，lead 就不準了。
        ceiling = int(longest * (1.0 - second)) if second >= 0 else None
        distance = edit_distance(read, combined, ceiling)
        score = 1.0 - distance / longest
        if score > best_score:
            second, second_text = best_score, (
                f"{best[0]} {best[1]}".strip() if best else ""
            )
            best_score = score
            best = (record.street, record.name)
        elif score > second:
            second, second_text = score, f"{record.street} {record.name}".strip()

    if best is None:
        return MapGuess(read=read)
    return MapGuess(
        street=best[0],
        name=best[1],
        score=best_score,
        runner_up=second_text,
        lead=max(0.0, best_score - second) if second >= 0 else 1.0,
        read=read,
    )
