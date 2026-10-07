"""速率估算。

兩個關鍵設計：

1. **x 軸用「活躍時間」而不是牆上時間。** 掛機、切頻、讀取畫面那段時間不會
   推進活躍時間，所以 EXP/hr 不會被稀釋成一個沒有意義的平均值。
2. **用線性回歸的斜率，不用首尾相減。** 首尾相減對單點雜訊與區間端點的
   選擇極度敏感；回歸把整個視窗的資訊都用上，雜訊會互相抵銷。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateEstimate:
    """某個時間窗的速率估計。"""

    window_sec: int
    exp_per_hour: float | None
    samples: int
    span_sec: float
    exp_gained: int

    @property
    def valid(self) -> bool:
        return self.exp_per_hour is not None


def linregress_slope(xs: list[float], ys: list[float]) -> float | None:
    """最小平方法的斜率 (dy/dx)。資料不足或 x 無變異時回傳 None。"""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    if sxx <= 0.0:
        return None
    return sxy / sxx


class ExpSeries:
    """(活躍時間, 累積經驗) 取樣點序列。

    只保留最長時間窗所需的點，記憶體用量有上界。
    """

    def __init__(self, windows: list[int]) -> None:
        self._windows = sorted({int(w) for w in windows if int(w) > 0}) or [300]
        # 多留一點緩衝，讓最長視窗邊界附近的回歸不會因為剛好被裁掉而失準。
        self._retain = self._windows[-1] * 1.2 + 60.0
        self._points: deque[tuple[float, int]] = deque()

    @property
    def windows(self) -> list[int]:
        return list(self._windows)

    def clear(self) -> None:
        self._points.clear()

    def add(self, active_t: float, cumulative_exp: int) -> None:
        """加入一個取樣點。

        活躍時間沒有前進時不會新增點，只更新最後一點：那代表「時鐘停了」
        （閒置或中斷），沒有新資訊。硬塞一堆 x 相同的點進去，回歸會被那一疊
        重複點拉著走，速率估計會莫名其妙地往下掉。
        """
        if self._points and active_t <= self._points[-1][0]:
            self._points[-1] = (self._points[-1][0], cumulative_exp)
            return
        self._points.append((active_t, cumulative_exp))
        self._prune(active_t)

    def _prune(self, now_active_t: float) -> None:
        cutoff = now_active_t - self._retain
        while len(self._points) > 2 and self._points[0][0] < cutoff:
            self._points.popleft()

    def __len__(self) -> int:
        return len(self._points)

    def rate(self, window_sec: int, now_active_t: float | None = None) -> RateEstimate:
        """估算指定時間窗內的 EXP/hr。"""
        if not self._points:
            return RateEstimate(window_sec, None, 0, 0.0, 0)

        now = self._points[-1][0] if now_active_t is None else now_active_t
        cutoff = now - window_sec
        xs: list[float] = []
        ys: list[float] = []
        for active_t, cum in self._points:
            if active_t >= cutoff:
                xs.append(active_t)
                ys.append(float(cum))

        if len(xs) < 2:
            return RateEstimate(window_sec, None, len(xs), 0.0, 0)

        span = xs[-1] - xs[0]
        gained = int(ys[-1] - ys[0])
        slope = linregress_slope(xs, ys)
        if slope is None:
            return RateEstimate(window_sec, None, len(xs), span, gained)
        # 斜率可能因雜訊略為負值；夾到 0 比顯示負速率有意義。
        per_hour = max(0.0, slope * 3600.0)
        return RateEstimate(window_sec, per_hour, len(xs), span, gained)

    def rates(self, now_active_t: float | None = None) -> dict[int, RateEstimate]:
        return {w: self.rate(w, now_active_t) for w in self._windows}


def eta_seconds(remaining_exp: int | None, exp_per_hour: float | None) -> float | None:
    """升級剩餘時間（秒）。資料不足或速率為零時回傳 None 而不是無限大。"""
    if remaining_exp is None or exp_per_hour is None:
        return None
    if remaining_exp <= 0:
        return 0.0
    if exp_per_hour <= 0.0:
        return None
    return remaining_exp / (exp_per_hour / 3600.0)


def format_duration(seconds: float | None) -> str:
    """把秒數格式化成 ``1h23m`` / ``12m34s`` / ``45s``。"""
    if seconds is None:
        return "--"
    if seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_elapsed(seconds: float | None) -> str:
    """經過時間，到小時仍然保留秒數（``1h23m45s``）。

    跟 :func:`format_duration` 分開是刻意的：活躍／閒置時間是「實際累積了多少」，
    秒數是真實資訊；而升級倒數那種**估計值**顯示到秒只是假精度，報表的欄寬也
    是照原本格式對齊的。兩種用途的取捨不同，不要共用一個函式。
    """
    if seconds is None or seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_exp(value: float | None) -> str:
    """大數字加上千分位；None 顯示為 ``--``。"""
    if value is None:
        return "--"
    return f"{int(round(value)):,}"


def format_rate(exp_per_hour: float | None) -> str:
    """速率。

    刻意用完整數字加千分位，不用 ``k``／``M`` 縮寫：縮寫看起來精簡，但要比較
    「1.2M/h 跟 980k/h 差多少」時腦袋得先換算一次，反而更慢。
    """
    if exp_per_hour is None:
        return "--"
    return f"{int(round(exp_per_hour)):,}/h"


def format_window(seconds: int | float | None) -> str:
    """時間窗的說法。給人看的標籤要講「15 分鐘」，不是「15m00s」。"""
    if seconds is None:
        return "--"
    total = int(round(seconds))
    if total >= 3600:
        hours = total / 3600
        return f"{hours:.0f} 小時" if abs(hours - round(hours)) < 0.05 else f"{hours:.1f} 小時"
    if total >= 60:
        return f"{total // 60} 分鐘"
    return f"{total} 秒"
