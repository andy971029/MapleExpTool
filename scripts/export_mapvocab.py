"""把客戶端的官方地圖清單匯出成 ``web/mapvocab.json``，給網頁版比對地圖名用。

桌面版每次啟動直接讀遊戲安裝目錄（見 ``gamedata.py``）；瀏覽器沒有權限讀本機
檔案，所以網頁版附一份快照。快照會隨遊戲改版過期 —— 改版後在裝了遊戲的機器上
重跑這支腳本、commit 新的 JSON 即可。格式是 ``[[map_id, street, name], ...]``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mapleexp import gamedata  # noqa: E402

OUT = ROOT / "web" / "mapvocab.json"


def main() -> int:
    vocab = gamedata.sync()
    if not vocab:
        print("找不到客戶端的地圖清單（遊戲沒裝，或 lz4 沒裝）", file=sys.stderr)
        return 1
    rows = [[r.map_id, r.street, r.name] for r in vocab.records]
    OUT.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT}: {len(rows)} 張地圖（來源 {vocab.source}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
