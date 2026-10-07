"""在畫面上自動找出經驗值欄位。

有了這個，第一次執行就不需要任何校準：程式自己掃一次畫面、認出
``113961[26.61%]`` 那一串在哪、把 ROI 與二值化參數設好。

找法是**先找錨點字元，再往兩邊展開**：

1. 只看畫面下方（狀態列一定在那裡）。
2. 用 ``%`` 的模板在整塊區域做二維比對。整條狀態列只有經驗值欄位有百分比符號
   —— HP/MP 雖然也是 ``[1446/2110]`` 這種括號格式，但裡面是斜線。
3. 在每個 ``%`` 候選位置切出一條窄窄的橫帶，用現有的切割 + 比對讀成字串。
4. 從 ``%`` 往右找 ``]``、往左找 ``[``，再往左吃掉括號前面那串數字。

一開始試過用「找出沒有 ink 的空白橫列來切出文字列」，在合成畫面上可行，
但在真實遊戲畫面上完全失效：1920 像素寬的任何一列都一定有東西（背景、
聊天視窗、角色），根本不存在空白列。改用錨點比對就沒有這個假設。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_ANCHOR, Roi
from ..core.reading import parse_exp_text
from .preprocess import binarize, suggest_binarization
from .segment import segment
from .templates import TemplateSet

# 狀態列在畫面下方這個比例的範圍內。
SEARCH_BAND_RATIO = 0.3

# 錨點字元比對的相似度下限。
ANCHOR_MIN_IOU = 0.82

# 每個門檻最多檢查幾個候選位置（由相似度高到低）。
MAX_CANDIDATES = 6

# 以錨點為中心切出來的橫帶，上下各留幾列。
# 這個值要夠小：狀態列的文字上方就是面板邊框亮線、下方就是經驗條，
# margin 一大就會把它們framed進來，整條變成一團切不開的東西。
# 括號比百分比符號高一列，所以 2 就夠了。
ROW_MARGIN = 2

# 錨點左右要看多遠（經驗值數字在左邊，收尾的 ``]`` 在右邊）。
# 左邊不要拉太遠，免得吃進 MP 的數字與長條圖。
LEFT_REACH = 160
RIGHT_REACH = 30

# 要嘗試的亮度門檻。自動建議值會排在最前面。
CANDIDATE_THRESHOLDS = (126, 110, 150, 170, 190)

# ROI 四周多留一點邊，免得抗鋸齒邊緣被切掉。
PADDING_X = 2
PADDING_Y = 1


@dataclass
class LocateResult:
    roi: Roi
    threshold: int
    invert: bool
    text: str
    exp_abs: int
    exp_pct: float | None
    rect: tuple[int, int, int, int]  # client 座標 (left, top, right, bottom)

    @property
    def confident(self) -> bool:
        return self.exp_pct is not None


def locate_exp_field(
    client_bgra: np.ndarray,
    templates: TemplateSet,
    anchor: str = DEFAULT_ANCHOR,
) -> LocateResult | None:
    """在 client area 畫面中找出經驗值欄位。找不到回傳 None。"""
    height, width = client_bgra.shape[:2]
    if height < 40 or width < 80:
        return None

    anchor_masks = templates.masks_for("%")
    if not anchor_masks:
        return None

    band_top = int(height * (1.0 - SEARCH_BAND_RATIO))
    band = client_bgra[band_top:, :]

    suggested, suggested_invert = suggest_binarization(band)
    attempts: list[tuple[int, bool]] = [(suggested, suggested_invert)]
    attempts.extend((t, False) for t in CANDIDATE_THRESHOLDS if t != suggested)

    fallback: LocateResult | None = None
    for threshold, invert in attempts:
        mask = binarize(band, threshold=threshold, invert=invert)
        for anchor_mask in anchor_masks:
            for anchor_y, anchor_x in _find_anchors(mask, anchor_mask):
                found = _read_around_anchor(
                    mask,
                    templates,
                    anchor_y=anchor_y,
                    anchor_x=anchor_x,
                    anchor_height=anchor_mask.shape[0],
                    anchor_width=anchor_mask.shape[1],
                    band_top=band_top,
                    width=width,
                    height=height,
                    threshold=threshold,
                    invert=invert,
                    anchor=anchor,
                )
                if found is None:
                    continue
                if found.confident:
                    return found
                if fallback is None:
                    fallback = found
    return fallback


def _find_anchors(mask: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    """在遮罩中找出與 template 相似的位置，由相似度高到低。

    用 IoU 而不是單純的相同像素數：後者會讓「一整塊實心區域」拿到高分。
    """
    th, tw = template.shape
    mh, mw = mask.shape
    if mh < th or mw < tw:
        return []

    source = mask.astype(np.int32)
    out_h, out_w = mh - th + 1, mw - tw + 1

    # 交集：只累加模板有 ink 的位移。
    intersection = np.zeros((out_h, out_w), dtype=np.int32)
    for dy in range(th):
        for dx in range(tw):
            if template[dy, dx]:
                intersection += source[dy : dy + out_h, dx : dx + out_w]

    # 視窗內的 ink 總數用積分圖一次算完。
    integral = np.zeros((mh + 1, mw + 1), dtype=np.int32)
    np.cumsum(np.cumsum(source, axis=0), axis=1, out=integral[1:, 1:])
    window = (
        integral[th : th + out_h, tw : tw + out_w]
        - integral[0:out_h, tw : tw + out_w]
        - integral[th : th + out_h, 0:out_w]
        + integral[0:out_h, 0:out_w]
    )

    template_ink = int(np.count_nonzero(template))
    union = window + template_ink - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)

    ys, xs = np.nonzero(iou >= ANCHOR_MIN_IOU)
    if ys.size == 0:
        return []
    scores = iou[ys, xs]
    order = np.argsort(-scores)

    picked: list[tuple[int, int]] = []
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        # 同一個符號會在相鄰位置重複命中，只留一個。
        if any(abs(y - py) < th and abs(x - px) < tw for py, px in picked):
            continue
        picked.append((y, x))
        if len(picked) >= MAX_CANDIDATES:
            break
    return picked


def _read_around_anchor(
    mask: np.ndarray,
    templates: TemplateSet,
    anchor_y: int,
    anchor_x: int,
    anchor_height: int,
    anchor_width: int,
    band_top: int,
    width: int,
    height: int,
    threshold: int,
    invert: bool,
    anchor: str,
) -> LocateResult | None:
    top = max(0, anchor_y - ROW_MARGIN)
    bottom = min(mask.shape[0], anchor_y + anchor_height + ROW_MARGIN)
    left = max(0, anchor_x - LEFT_REACH)
    right = min(mask.shape[1], anchor_x + anchor_width + RIGHT_REACH)

    strip = mask[top:bottom, left:right]
    glyphs = segment(strip, expected_width=templates.median_width())
    if len(glyphs) < 4:
        return None

    labels = [templates.match(glyph.mask).char or "?" for glyph in glyphs]

    # 欄位之間的空隙比字元之間大得多。把「哪些字元前面有大空隙」算出來，
    # 往左吃數字時才不會跨過空隙把隔壁 MP 的數字也吃進來。
    max_gap = max(3.0, (templates.median_width() or 5.0) * 1.2)
    breaks = {
        index
        for index in range(1, len(glyphs))
        if glyphs[index].left - glyphs[index - 1].right > max_gap
    }
    span = _exp_span(labels, breaks)
    if span is None:
        return None
    start, end = span

    text = "".join(labels[start:end])
    exp_abs, exp_pct = parse_exp_text(text)
    if exp_abs is None:
        return None

    chosen = glyphs[start:end]
    rect = (
        max(0, left + min(g.left for g in chosen) - PADDING_X),
        max(0, band_top + top + min(g.top for g in chosen) - PADDING_Y),
        min(width, left + max(g.right for g in chosen) + PADDING_X),
        min(height, band_top + top + max(g.bottom for g in chosen) + PADDING_Y),
    )
    if rect[2] - rect[0] < 8 or rect[3] - rect[1] < 5:
        return None

    return LocateResult(
        roi=Roi.from_pixels(rect, width, height, anchor=anchor),
        threshold=threshold,
        invert=invert,
        text=text,
        exp_abs=exp_abs,
        exp_pct=exp_pct,
        rect=rect,
    )


def _exp_span(
    labels: list[str], breaks: set[int] | None = None
) -> tuple[int, int] | None:
    """在一列辨識結果裡找出經驗值欄位的字元範圍 [start, end)。

    ``breaks`` 是「前面有明顯空隙」的字元索引集合。沒有它的話，往左吃數字會
    一路吃過欄位之間的空隙，把隔壁 MP 的數字也當成經驗值的一部分。
    """
    breaks = breaks or set()
    try:
        percent = labels.index("%")
    except ValueError:
        return None

    # 往右找收尾的 ``]``。
    end = percent + 1
    while end < len(labels) and labels[end] != "]":
        if end in breaks or end - percent > 3:
            return None
        end += 1
    end = min(len(labels), end + 1)

    # 往左找 ``[``。
    open_bracket = percent
    while open_bracket >= 0 and labels[open_bracket] != "[":
        if open_bracket in breaks or percent - open_bracket > 12:
            return None
        open_bracket -= 1
    if open_bracket < 0:
        return None

    # 再往左吃掉括號前面那串數字，遇到空隙就停。
    start = open_bracket
    while start > 0 and labels[start - 1].isdigit() and start not in breaks:
        start -= 1
    if start == open_bracket:
        return None  # 括號前面沒有數字，那不是經驗值欄位
    return start, end
