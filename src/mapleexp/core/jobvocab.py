"""把 OCR 讀到的職業名稱對到已知的職業清單。

職業是封閉集合，而狀態列上的字很小 —— 資料庫裡就有 ``槍騎兵`` 被讀成 ``搶騎兵``
的實際紀錄。有清單的話，一個字的差距靠編輯距離就能修回來，做法跟地圖名一樣
（見 :mod:`mapvocab`）。

**清單為什麼是內建的、不是從客戶端讀。** 翻過客戶端的 json bundle（71 張 TextAsset
表），沒有一張「職業表」。職業名稱只以三種形式存在：

1. NPC 腳本的字串表裡一串**沒有分隔**的名字（``初心者劍士狂戰士十字軍…``），
   容器是被 IL2CPP 剝掉 type tree 的 MonoBehaviour，位移表另外放，要解得先搞懂
   它的序列化版面；
2. ``SkillBook`` 表的技能書名（``槍騎兵之路``、``光之聖騎士``），不是乾淨的職業名；
3. NPC 對話文字。

都不是能直接拿來比對的東西。而職業清單幾年才動一次（新職業上線），跟地圖清單
「每次改版都增加」的性質不同，內建一份是合理的取捨。下面這份是照 (1) 那串字、
依職業 ID 順序抄下來，再依狀態列實際顯示修正的：客戶端那串漏了弓箭手一轉的
``弓箭手``；法師二轉在那串裡寫 ``巫師(火/毒)``，狀態列顯示的是 ``巫師(火、毒)``，
以狀態列為準（比對時標點會被清掉，兩種寫法都配得上，差別只在存進資料庫的寫法）。

跟地圖比對不同的是**有門檻**：地圖一定會挑一張，因為指紋才是鍵、名字只是給人看；
職業沒有指紋，讀到不像任何職業的東西（例如職業跟角色名那兩行被切反了）時，
保留原始 OCR 結果比硬配一個職業誠實。
"""

from __future__ import annotations

from dataclasses import dataclass

from .mapvocab import clean, similarity

JOB_NAMES: tuple[str, ...] = (
    "初心者",
    "劍士", "狂戰士", "十字軍", "英雄",
    "見習騎士", "騎士", "聖騎士",
    "槍騎兵", "龍騎士", "黑騎士",
    "法師",
    "巫師(火、毒)", "魔導士(火、毒)", "大魔導士(火、毒)",
    "巫師(冰、雷)", "魔導士(冰、雷)", "大魔導士(冰、雷)",
    "僧侶", "祭司", "主教",
    "弓箭手", "獵人", "遊俠", "箭神",
    "弩弓手", "狙擊手", "神射手",
    "盜賊", "刺客", "暗殺者", "夜使者",
    "俠盜", "神偷", "暗影神偷",
    "海盜", "打手", "格鬥家", "拳霸",
    "槍手", "神槍手", "槍神",
    "貴族", "聖魂劍士", "烈焰巫師", "破風使者", "暗夜行者", "閃雷悍將",
    "傳說", "狂狼勇士",
)

# 相似度低於這個值就不採用清單裡的名字，保留 OCR 原文。
# 三個字錯一個是 0.67、兩個字錯一個是 0.5，所以 0.5 剛好把「錯一個字」收進來、
# 把「整個不像」擋在外面。
MIN_SCORE = 0.5

# OCR 常把「槍」看成的字 -> 「槍」。做法跟 mapvocab 的羅馬數字形近字一樣：
# 清單裡沒有任何職業含這些字，直接換掉不會誤傷。
#
# 為什麼不能只靠編輯距離：``搶手`` 跟 ``槍手``、``打手`` 都只差一個字，兩個字的
# 名字錯一個就是 0.5 同分，而 ``打手`` 在清單裡排得比較前面，於是槍手玩家被判成
# 打手（使用者實際回報）。``搶騎兵`` 沒出事只是因為三個字的職業裡沒有別的
# ``?騎兵``。形近字先換掉，``搶手`` 就是 ``槍手`` 的完全命中，不用靠同分抽籤。
_LOOKALIKES = str.maketrans({
    "搶": "槍",     # 資料庫裡的實際紀錄（搶騎兵、搶手）
    "鎗": "槍",     # 槍的異體字
    "抢": "槍",     # 簡體
})


@dataclass(frozen=True)
class JobGuess:
    """比對結果。``name`` 是要顯示／儲存的字：配得上就是清單裡的寫法，否則是 OCR 原文。"""

    name: str
    score: float
    read: str
    """OCR 讀到的（清理過的）文字，對照用。"""
    matched: bool = False
    """``name`` 是不是清單裡的職業。不能從 ``score`` 推：兩個職業同分時分數夠高但
    仍然沒有採用。"""


def identify(text: str) -> JobGuess:
    """挑出清單裡跟 OCR 結果最接近的職業；差太多、或分不出高下就原樣保留。

    **同分不猜。** 讀到只剩一個 ``手`` 字時，``打手`` 跟 ``槍手`` 都是 0.5，沒有任何
    依據挑哪一個；照清單順序挑等於擲硬幣，而猜錯會把整段統計記在別的職業底下。
    這時回報「配不上」，呼叫端會保留上一次讀對的結果（跟模板比對「前兩名差距太小
    就讓這一格失敗」是同一個原則）。
    """
    read = clean(text).translate(_LOOKALIKES)
    if not read:
        return JobGuess(name="", score=0.0, read=read)

    best_name, best_score, runner_up = "", -1.0, -1.0
    for job in JOB_NAMES:
        score = similarity(read, clean(job))
        if score > best_score:
            best_name, best_score, runner_up = job, score, best_score
        elif score > runner_up:
            runner_up = score

    if best_score < MIN_SCORE or best_score == runner_up:
        return JobGuess(name=(text or "").strip(), score=best_score, read=read)
    return JobGuess(name=best_name, score=best_score, read=read, matched=True)
