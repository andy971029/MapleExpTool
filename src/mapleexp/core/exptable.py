"""等級經驗表：每個等級升到下一級需要多少經驗。

設計決策：**不內建硬寫的經驗表。**

理由是狀態列同時顯示絕對經驗值與百分比，兩者相除就能即時反推本級所需經驗，
精度由百分比的小數位數決定（兩位小數 => 在 30% 時誤差約 0.03%）。
自己推導比內建一張可能記錯的表可靠得多，而且改版調整經驗曲線也不會失效。

推導方式用「區間交集」而不是取平均：
每次觀測給出一個 N 的可能區間，真值必定落在所有區間的交集內，
所以多觀測幾次就會單調收斂，而且不會被平均值的偏差帶跑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 區間相對寬度小於此值才視為「可信」，可用於升級接續與升級倒數。
RELIABLE_REL_WIDTH = 0.02

# 已經累積這麼多次觀測後，與現有區間衝突的新觀測視為辨識錯誤而忽略，
# 而不是推翻既有結果。
ESTABLISHED_OBSERVATIONS = 3


@dataclass
class LevelNeed:
    """某個等級的「所需經驗」估計狀態。"""

    lo: float
    hi: float
    observations: int = 1
    source: str = "derived"  # "derived" | "manual"

    @property
    def value(self) -> int:
        if self.source == "manual":
            return int(round(self.lo))
        return int(round((self.lo + self.hi) / 2.0))

    @property
    def rel_width(self) -> float:
        if self.lo <= 0:
            return float("inf")
        return (self.hi - self.lo) / self.lo

    @property
    def reliable(self) -> bool:
        return self.source == "manual" or self.rel_width <= RELIABLE_REL_WIDTH


class ExpTable:
    """等級 -> 所需經驗。可持久化，跨場次累積精度。"""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._needs: dict[int, LevelNeed] = {}
        self._dirty = False
        if path is not None and path.exists():
            self.load(path)

    # ------------------------------------------------------------------ #
    # 觀測
    # ------------------------------------------------------------------ #

    def observe_bounds(self, level: int | None, bounds: tuple[float, float] | None) -> bool:
        """餵入一次觀測區間。回傳是否實際更新了估計值。

        ``level`` 為 None 時（沒有校準等級 ROI）直接忽略 —— 追蹤器仍然可以
        使用單筆取樣自己推導的 need，只是無法跨場次累積。
        """
        if level is None or bounds is None:
            return False
        lo, hi = bounds
        if not (0 < lo <= hi):
            return False

        current = self._needs.get(level)
        if current is None:
            self._needs[level] = LevelNeed(lo=lo, hi=hi)
            self._dirty = True
            return True

        if current.source == "manual":
            # 手動指定的值視為權威，不被觀測覆蓋。
            return False

        new_lo = max(current.lo, lo)
        new_hi = min(current.hi, hi)
        if new_lo > new_hi:
            # 交集為空：其中一邊是辨識錯誤。
            if current.observations >= ESTABLISHED_OBSERVATIONS:
                return False  # 既有估計已有足夠支撐，忽略這筆異常觀測
            self._needs[level] = LevelNeed(lo=lo, hi=hi)
            self._dirty = True
            return True

        if new_lo == current.lo and new_hi == current.hi:
            current.observations += 1
            return False

        current.lo = new_lo
        current.hi = new_hi
        current.observations += 1
        self._dirty = True
        return True

    def set_manual(self, level: int, need: int) -> None:
        """手動指定某等級的所需經驗（例如從官方資料抄來的值）。"""
        self._needs[level] = LevelNeed(
            lo=float(need), hi=float(need), observations=1, source="manual"
        )
        self._dirty = True

    # ------------------------------------------------------------------ #
    # 查詢
    # ------------------------------------------------------------------ #

    def need(self, level: int | None) -> int | None:
        if level is None:
            return None
        entry = self._needs.get(level)
        return entry.value if entry is not None else None

    def reliable_need(self, level: int | None) -> int | None:
        """只在估計足夠精確時回傳值，否則 None。"""
        if level is None:
            return None
        entry = self._needs.get(level)
        if entry is None or not entry.reliable:
            return None
        return entry.value

    def confidence(self, level: int | None) -> str:
        if level is None:
            return "unknown"
        entry = self._needs.get(level)
        if entry is None:
            return "unknown"
        if entry.source == "manual":
            return "manual"
        return "reliable" if entry.reliable else "rough"

    def known_levels(self) -> list[int]:
        return sorted(self._needs)

    def __len__(self) -> int:
        return len(self._needs)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #

    def load(self, path: Path | None = None) -> None:
        target = path or self._path
        if target is None or not target.exists():
            return
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        needs = raw.get("needs") if isinstance(raw, dict) else None
        if not isinstance(needs, dict):
            return
        for key, value in needs.items():
            try:
                level = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, (int, float)):
                # 簡寫格式：{"30": 98280} —— 視為手動指定值。
                self._needs[level] = LevelNeed(
                    lo=float(value), hi=float(value), source="manual"
                )
                continue
            if not isinstance(value, dict):
                continue
            try:
                self._needs[level] = LevelNeed(
                    lo=float(value["lo"]),
                    hi=float(value["hi"]),
                    observations=int(value.get("observations", 1)),
                    source=str(value.get("source", "derived")),
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._dirty = False

    def save(self, path: Path | None = None, force: bool = False) -> Path | None:
        target = path or self._path
        if target is None:
            return None
        if not self._dirty and not force:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "自動推導的等級經驗表；lo/hi 為可能區間，交集越窄越準。",
            "needs": {
                str(level): {
                    "lo": round(entry.lo, 2),
                    "hi": round(entry.hi, 2),
                    "value": entry.value,
                    "observations": entry.observations,
                    "source": entry.source,
                }
                for level, entry in sorted(self._needs.items())
            },
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        self._dirty = False
        return target
