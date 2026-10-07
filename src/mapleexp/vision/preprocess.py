"""把擷取到的 BGRA 影像變成二值遮罩。

輸出的遮罩裡 ``True`` 代表「文字像素」（下面一律稱為 ink）。
後面的切割與比對都只看這個遮罩，所以整條管線對顏色、亮度的變化只需要
在這一層處理。
"""

from __future__ import annotations

import numpy as np

# 亮度係數（BT.601，整數近似）。用整數是為了避免不必要的浮點轉換。
_LUMA_B = 29
_LUMA_G = 150
_LUMA_R = 77


def to_luma(bgra: np.ndarray) -> np.ndarray:
    """BGRA -> 灰階 (uint8)。"""
    if bgra.ndim != 3 or bgra.shape[2] < 3:
        raise ValueError("預期為 (h, w, 3+) 的影像")
    pixels = bgra[..., :3].astype(np.uint32)
    luma = (
        pixels[..., 0] * _LUMA_B + pixels[..., 1] * _LUMA_G + pixels[..., 2] * _LUMA_R
    ) >> 8
    return luma.astype(np.uint8)


def binarize(
    bgra: np.ndarray,
    threshold: int = 150,
    invert: bool = False,
    ink_color: list[int] | tuple[int, int, int] | None = None,
    color_tolerance: int = 90,
) -> np.ndarray:
    """產生 ink 遮罩。

    有給 ``ink_color``（BGR）時走色鍵：計算每個像素與目標色的歐氏距離，
    距離在容許範圍內的算 ink。狀態列背景有紋理或漸層時，色鍵比亮度門檻乾淨得多。

    沒給顏色時走亮度門檻：``invert=False`` 取亮字（暗底），``True`` 取暗字（亮底）。
    """
    if ink_color is not None and len(ink_color) >= 3:
        target = np.array(ink_color[:3], dtype=np.int32).reshape(1, 1, 3)
        diff = bgra[..., :3].astype(np.int32) - target
        distance_sq = (diff * diff).sum(axis=2)
        return distance_sq <= int(color_tolerance) * int(color_tolerance)

    luma = to_luma(bgra)
    if invert:
        return luma <= np.uint8(threshold)
    return luma >= np.uint8(threshold)


def otsu_threshold(luma: np.ndarray) -> int:
    """Otsu 自動門檻，給校準介面當起始建議值。

    回傳值的慣例配合 :func:`binarize`：``luma >= 回傳值`` 就是前景。
    教科書的 Otsu 給的是「背景類別的最大值 t」（前景為 ``> t``），
    所以這裡回傳 ``t + 1`` —— 少了這個 +1，在雙峰分佈上會把整張圖都判成前景。
    """
    if luma.size == 0:
        return 128
    histogram = np.bincount(luma.reshape(-1), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 128

    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(histogram * levels)
    sum_total = sum_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return 128

    mean_bg = np.zeros(256)
    mean_fg = np.zeros(256)
    np.divide(sum_bg, weight_bg, out=mean_bg, where=weight_bg > 0)
    np.divide(sum_total - sum_bg, weight_fg, out=mean_fg, where=weight_fg > 0)

    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return min(255, int(np.argmax(between)) + 1)


def _glyph_like_runs(mask: np.ndarray, min_width: int = 2) -> int:
    """遮罩裡有多少個「寬度足夠、像字元」的垂直區塊。

    這是判斷一組二值化參數好不好的關鍵指標：切出一串分開的小區塊，
    比切出一大坨或一片空白更像文字。
    """
    if mask.size == 0:
        return 0
    occupied = mask.any(axis=0)
    count = 0
    run = 0
    for value in occupied:
        if value:
            run += 1
        else:
            if run >= min_width:
                count += 1
            run = 0
    if run >= min_width:
        count += 1
    return count


def suggest_binarization(
    bgra: np.ndarray, target_ink: float = 0.18
) -> tuple[int, bool]:
    """猜一組 (threshold, invert)。

    不直接用 Otsu：狀態列的面板背景常常是漸層，Otsu 會把「背景暗的部分」與
    「背景亮的部分」分開，而不是把文字跟背景分開 —— 結果是整條 ROI 變成
    一個連通區塊，一個字元都切不出來。

    改用的判準貼近實際目標：
    1. 文字只佔少數像素，所以 ink 比例要落在合理區間；
    2. 好的門檻會切出「一串分開的小區塊」，壞的門檻切出一大坨。
    直接拿區塊數當分數，再用 ink 比例接近目標值當 tie-break。
    """
    luma = to_luma(bgra)
    best: tuple[int, float, int, bool] | None = None

    for invert in (False, True):
        for threshold in range(8, 252, 2):
            mask = (luma <= threshold) if invert else (luma >= threshold)
            ratio = float(mask.mean())
            if not (0.04 <= ratio <= 0.42):
                continue
            mask = despeckle(mask)
            runs = _glyph_like_runs(mask)
            if runs == 0:
                continue
            candidate = (runs, -abs(ratio - target_ink), threshold, invert)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        # 沒有任何門檻切得出像文字的東西，退回 Otsu 讓使用者有個起點。
        threshold = otsu_threshold(luma)
        return threshold, float((luma >= threshold).mean()) > 0.5
    return best[2], best[3]


def despeckle(mask: np.ndarray) -> np.ndarray:
    """移除孤立的單一像素。

    抗鋸齒邊緣與畫面壓縮會留下一堆單點雜訊。它們本身不是字元，但只要黏在
    字元的欄位範圍內，就會把那個字元的外接矩形撐大、害模板比對失配。
    判斷方式是「八方相鄰完全沒有其他 ink」。
    """
    if mask.size == 0 or not mask.any():
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    neighbours = np.zeros((height, width), dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            neighbours += padded[dy : dy + height, dx : dx + width]
    return mask & (neighbours > 0)


def ink_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """ink 的外接矩形 (left, top, right, bottom)；完全沒有 ink 時回傳 None。"""
    if mask.size == 0 or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def trim(mask: np.ndarray) -> np.ndarray:
    """裁掉遮罩四周沒有 ink 的空白。"""
    bounds = ink_bounds(mask)
    if bounds is None:
        return mask[:0, :0]
    left, top, right, bottom = bounds
    return mask[top:bottom, left:right]


def ink_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())
