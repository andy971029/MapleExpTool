"""在真實的客戶端清單上，量地圖名比對的錯誤率。

比對只有一個判斷：OCR 讀到的整串文字跟每張地圖的「地區＋地圖名」算編輯距離，
挑最接近的，沒有門檻。這支腳本用模擬的 OCR 缺字／誤認去估它的錯誤率 —— 真正的
數字還是要靠實際使用，這裡只是先有個底。

用法::

    python scripts/eval_mapvocab.py
    python scripts/eval_mapvocab.py --trials 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapleexp import gamedata  # noqa: E402
from mapleexp.core import mapvocab  # noqa: E402

# OCR 把筆畫黏在一起時常吐出來的字 —— 誤認成這些比誤認成隨機字更接近真實。
CONFUSABLE = "的同圖國之地上下山村林洞口大小中日目月田回品"

# (標籤, 保留率, 誤認率)。「乾淨」對應前處理改成解混合之後的實際品質：實機連續
# 18 幀都一字不差；後面幾組是留著看退化時會壞成什麼樣。
CONDITIONS = (
    ("乾淨", 0.98, 0.01),
    ("小缺字", 0.85, 0.05),
    ("中等", 0.60, 0.10),
    ("嚴重", 0.40, 0.20),
)


def degrade(text: str, keep: float, swap: float, rng: random.Random) -> str:
    """模擬 OCR：以機率 *keep* 保留每個字，保留下來的以機率 *swap* 讀錯。"""
    out = []
    for char in text:
        if rng.random() > keep:
            continue
        out.append(rng.choice(CONFUSABLE) if rng.random() < swap else char)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20251005)
    parser.add_argument("--show", type=int, default=8, help="列出幾個答錯的例子")
    args = parser.parse_args()

    vocab = gamedata.sync()
    if not vocab:
        print("找不到客戶端的地圖資料。", file=sys.stderr)
        return 2
    print(f"清單：{len(vocab)} 張地圖、{len(vocab.names)} 個不重複名稱、"
          f"{len(vocab.streets)} 個地區\n")

    rng = random.Random(args.seed)
    print(f"{'條件':>6} {'樣本':>7} {'答對':>7} {'給不出答案':>11} {'硬猜(lead=0)':>13}")
    print("-" * 50)
    mistakes = []
    for label, keep, swap in CONDITIONS:
        total = correct = blank = tied = 0
        for record in vocab.records:
            if not record.name:
                continue
            for _ in range(args.trials):
                read = degrade(record.street, keep, swap, rng) + degrade(
                    record.name, keep, swap, rng
                )
                guess = mapvocab.identify(read, vocab)
                total += 1
                if not guess.name:
                    blank += 1
                    continue
                if guess.lead <= 0.0:
                    tied += 1
                if (guess.street, guess.name) == (record.street, record.name):
                    correct += 1
                elif label == "乾淨" and len(mistakes) < args.show:
                    mistakes.append((read, guess, record))
        print(
            f"{label:>6} {total:>7,} {correct / total:>6.1%} "
            f"{blank / total:>10.1%} {tied / total:>12.1%}"
        )

    if mistakes:
        print("\n「乾淨」條件下答錯的例子：")
        for read, guess, record in mistakes:
            print(
                f"  讀到 {read!r}\n"
                f"     選了 {guess.full_name}（相似度 {guess.score:.2f}、"
                f"領先 {guess.lead:.2f}）\n"
                f"     正解 {record.street} {record.name}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
