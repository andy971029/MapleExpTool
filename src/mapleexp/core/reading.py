"""一次狀態列取樣的結果，以及把辨識出來的字串解析成數值的邏輯。

辨識層只負責「把 ROI 變成字串」，這裡負責「把字串變成意義」。
兩者分開的好處是：解析規則可以在沒有遊戲、沒有截圖的情況下完整測試。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 經驗值上限（防止把雜訊讀成天文數字）。
EXP_MAX = 10**13

# 百分比小於這個值時，用 exp/pct 推導「本級所需經驗」會太不精確，不採用。
MIN_PCT_FOR_DERIVATION = 0.5

# 顯示的百分比只有兩位小數，實際比例落在 ±0.01 之內
# （同時涵蓋四捨五入與直接截斷兩種可能的顯示方式）。
PCT_DISPLAY_TOLERANCE = 0.01


_DECIMAL_RE = re.compile(r"\d{1,3}\.\d{1,2}")
_DIGITS_RE = re.compile(r"\d+")
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


def _clean(raw: str) -> str:
    """只留下數字與有意義的符號。

    千分位逗號直接刪掉（它在數字內部），其他雜訊字元則換成空白而**不是刪掉** ——
    刪掉會把本來分開的數字串黏成一串（``"7 1234567"`` 變成 ``"71234567"``），
    也會讓小數點的比對跨越本來不相鄰的字元。
    """
    text = (raw or "").replace(",", "")
    text = re.sub(r"[^0-9.\[\]]+", " ", text)
    return text.strip()


def parse_exp_text(raw: str) -> tuple[int | None, float | None]:
    """把狀態列的經驗值字串解析成 (絕對經驗值, 百分比)。

    支援的版面（由明確到模糊）：

    1. ``12345[30.25%]``  -> 括號外是絕對值、括號內是百分比（楓之谷經典版面）
    2. ``12345 30.25``    -> 有小數點的是百分比，最長的整數串是絕對值
    3. ``12345``          -> 只有絕對值
    4. ``30.25``          -> 只有百分比

    任一項解析不出來就回傳 None，交給上層決定要不要當成失敗。
    """
    text = _clean(raw)
    if not text:
        return None, None

    exp_abs: int | None = None
    exp_pct: float | None = None

    bracket = _BRACKET_RE.search(text)
    if bracket is not None:
        # 版面 1：括號把兩個數字清楚分開，最可靠。
        inside = bracket.group(1)
        outside = text[: bracket.start()] + text[bracket.end() :]
        exp_pct = _first_float(inside)
        exp_abs = _longest_int(outside)
        if exp_pct is None:
            inside_int = _longest_int(inside)
            if inside_int is not None:
                if inside_int <= 100:
                    # 遊戲顯示整數百分比。
                    exp_pct = float(inside_int)
                elif inside_int < 10000:
                    # 小數點沒讀到（它只有 2x2 像素，很容易被濾掉或與鄰字黏住）。
                    # 這個遊戲的百分比固定兩位小數，所以把最後兩位還原成小數即可，
                    # 例如 2750 -> 27.50。少了這條，小數點一讀不到就整個百分比報廢。
                    exp_pct = inside_int / 100.0
    else:
        decimal = _DECIMAL_RE.search(text)
        if decimal is not None:
            exp_pct = _safe_float(decimal.group(0))
            remainder = text[: decimal.start()] + text[decimal.end() :]
            exp_abs = _longest_int(remainder)
        else:
            exp_abs = _longest_int(text)

    if exp_pct is not None and not (0.0 <= exp_pct <= 100.0):
        exp_pct = None
    if exp_abs is not None and not (0 <= exp_abs < EXP_MAX):
        exp_abs = None

    return exp_abs, exp_pct


def _longest_int(text: str) -> int | None:
    """取最長的連續數字串；長度相同時取先出現的那個。"""
    best: str | None = None
    for match in _DIGITS_RE.finditer(text):
        token = match.group(0)
        if best is None or len(token) > len(best):
            best = token
    if best is None:
        return None
    try:
        return int(best)
    except ValueError:
        return None


def _first_float(text: str) -> float | None:
    match = _DECIMAL_RE.search(text)
    return _safe_float(match.group(0)) if match else None


def _safe_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


@dataclass(frozen=True)
class StatusReading:
    """單次取樣。

    ``mono`` 用 :func:`time.perf_counter`，只用來算時間差（不受系統時間調整影響）；
    ``wall`` 用 :func:`time.time`，只用來存檔與顯示。
    兩個都存是因為兩者的用途不能互換。
    """

    mono: float
    wall: float
    ok: bool
    level: int | None = None
    exp_abs: int | None = None
    exp_pct: float | None = None
    raw_exp: str = ""
    reason: str = ""
    scores: tuple[float, ...] = field(default=(), repr=False)

    @classmethod
    def failed(cls, mono: float, wall: float, reason: str, raw_exp: str = "") -> "StatusReading":
        return cls(mono=mono, wall=wall, ok=False, reason=reason, raw_exp=raw_exp)

    @property
    def has_exp(self) -> bool:
        return self.exp_abs is not None

    @property
    def derived_need(self) -> int | None:
        """由 (絕對經驗值, 百分比) 反推「本級升級所需經驗」。

        這是整個設計的關鍵：有了它，就算完全沒有校準等級 ROI、也沒有任何
        內建等級經驗表，升級瞬間的經驗歸零也能正確接續。
        """
        bounds = self.need_bounds
        if bounds is None:
            return None
        lo, hi = bounds
        return int(round((lo + hi) / 2.0))

    @property
    def need_bounds(self) -> tuple[float, float] | None:
        """「本級所需經驗」的可能區間。

        顯示的百分比 d 代表真實比例落在 [d - tol, d + tol]，於是
        所需經驗 N = 100 * exp / ratio 落在 [100*exp/(d+tol), 100*exp/(d-tol)]。
        區間比單一估計值有用：多次觀測可以交集收斂（見 ExpTable）。
        """
        if self.exp_abs is None or self.exp_pct is None:
            return None
        if self.exp_pct < MIN_PCT_FOR_DERIVATION:
            return None
        if self.exp_abs <= 0:
            return None
        lo_pct = self.exp_pct - PCT_DISPLAY_TOLERANCE
        hi_pct = self.exp_pct + PCT_DISPLAY_TOLERANCE
        if lo_pct <= 0:
            return None
        lo = 100.0 * self.exp_abs / hi_pct
        hi = 100.0 * self.exp_abs / lo_pct
        return lo, hi
