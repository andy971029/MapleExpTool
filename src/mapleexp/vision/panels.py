"""通用的遊戲 UI 面板偵測。

楓之谷的每個功能視窗（小地圖、背包、技能欄、角色資料…）都可以被玩家拖到
任何位置，所以**不能寫死座標**。但它們的外框樣式是一致的：最上面一定有一條
**純白底、純黑字的標題列**（實機量過四個面板，底色都是 BGR(255,255,255)、
文字都是 BGR(0,0,0)）。

於是流程變成：

1. 找出畫面上所有「白色細長條」—— 這些是標題列的候選。
2. 對每一條做 OCR，得到標題文字。
3. 跟已知標題清單做**模糊比對**（OCR 會有一兩個字的誤差，像
   ``技能欄`` 讀成 ``技能攔``、``小地圖`` 讀成 ``小地``）。
4. 比對到的就是那個面板，標題列下方就是它的內容區。

這樣要支援新的面板，只要在 :data:`KNOWN_TITLES` 加一個字串。

為什麼不直接用「白色條」當判準就好：背包裡的楓幣／點數欄位也是白底黑字的
細長條（實測 OCR 讀到 ``128,231``）。靠標題文字比對就自然排除了 ——
它配不上任何已知標題。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ocr
from .segment import label_components

# 標題列：純白底。容許一點點壓縮雜訊。
WHITE_MIN = 230
WHITE_MAX_SPREAD = 12

# 標題列的合理尺寸與填充率。
TITLE_MIN_WIDTH = 40
TITLE_MAX_WIDTH = 700
TITLE_MIN_HEIGHT = 10
TITLE_MAX_HEIGHT = 30
# 「小地圖」只有三個字，黑字就吃掉快一半，所以填充率門檻不能訂高。
TITLE_MIN_FILL = 0.40

# 模糊比對的相似度下限，以及第一名要領先第二名多少。
MATCH_MIN_SCORE = 0.45
MATCH_MIN_MARGIN = 0.15

# 內容區：標題列底下這個範圍（相對於標題列）。
BODY_PAD_LEFT = 14
BODY_PAD_RIGHT = 90
BODY_HEIGHT = 170

KNOWN_TITLES: tuple[str, ...] = (
    "小地圖",
    "角色資料",
    "技能欄",
    "裝備欄",
    "ITEM INVENTORY",
    "任務指引",
)


@dataclass
class Panel:
    """畫面上的一個功能面板。"""

    title: str                      # 比對到的標準標題
    raw_title: str                  # OCR 實際讀到的字
    score: float
    title_rect: tuple[int, int, int, int]
    body_rect: tuple[int, int, int, int]
    white_fill: float = 0.0


def white_mask(frame: np.ndarray) -> np.ndarray:
    bgr = frame[..., :3].astype(np.int16)
    highest = bgr.max(axis=2)
    lowest = bgr.min(axis=2)
    return (lowest >= WHITE_MIN) & ((highest - lowest) <= WHITE_MAX_SPREAD)


def find_title_bars(frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """找出所有長得像標題列的白色細長條，由上而下排序。"""
    mask = white_mask(frame)
    if not mask.any():
        return []
    _labels, boxes = label_components(mask, connectivity=8)

    bars: list[tuple[tuple[int, int, int, int], float]] = []
    for left, top, right, bottom in boxes.values():
        width, height = right - left, bottom - top
        if not (TITLE_MIN_WIDTH <= width <= TITLE_MAX_WIDTH):
            continue
        if not (TITLE_MIN_HEIGHT <= height <= TITLE_MAX_HEIGHT):
            continue
        fill = float(mask[top:bottom, left:right].mean())
        if fill < TITLE_MIN_FILL:
            continue
        bars.append(((left, top, right, bottom), fill))
    bars.sort(key=lambda item: (item[0][1], item[0][0]))
    return bars


def similarity(a: str, b: str) -> float:
    """1 - 正規化編輯距離。用來吸收 OCR 的一兩個字誤差。"""
    a = (a or "").replace(" ", "").upper()
    b = (b or "").replace(" ", "").upper()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def match_title(text: str, titles: tuple[str, ...] = KNOWN_TITLES) -> tuple[str, float]:
    """把 OCR 結果配到已知標題。配不上回 ("", 分數)。"""
    scored = sorted(
        ((similarity(text, title), title) for title in titles), reverse=True
    )
    if not scored:
        return "", 0.0
    best_score, best_title = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < MATCH_MIN_SCORE or (best_score - second) < MATCH_MIN_MARGIN:
        return "", best_score
    return best_title, best_score


def find_panels(
    frame: np.ndarray, titles: tuple[str, ...] = KNOWN_TITLES
) -> list[Panel]:
    """找出畫面上所有認得的面板。

    每個候選標題列都要跑一次 OCR，所以這不便宜（一次大概幾百毫秒）。
    呼叫端應該快取結果，只在需要時重新找（見 :class:`PanelTracker`）。
    """
    height, width = frame.shape[:2]
    panels: list[Panel] = []
    for (left, top, right, bottom), fill in find_title_bars(frame):
        crop = frame[max(0, top - 2) : bottom + 2, max(0, left - 2) : right + 2]
        raw = ocr.recognize_best(crop, scales=(3, 4, 5), bright_text=False)
        raw = raw.replace("\n", "")
        title, score = match_title(raw, titles)
        if not title:
            continue
        panels.append(
            Panel(
                title=title,
                raw_title=raw,
                score=score,
                title_rect=(left, top, right, bottom),
                body_rect=(
                    max(0, left - BODY_PAD_LEFT),
                    bottom,
                    min(width, right + BODY_PAD_RIGHT),
                    min(height, bottom + BODY_HEIGHT),
                ),
                white_fill=fill,
            )
        )
    return panels


class PanelTracker:
    """記住找到的面板，只在需要時重新搜尋。

    面板搜尋要對每個候選標題列跑 OCR，太慢，不能每次取樣都做。
    玩家不會一直搬視窗，所以「找到就記住、用壞了再找」是對的取捨。
    """

    def __init__(self, titles: tuple[str, ...] = KNOWN_TITLES) -> None:
        self.titles = titles
        self.panels: dict[str, Panel] = {}
        self.searched = 0

    def get(self, title: str) -> Panel | None:
        return self.panels.get(title)

    def refresh(self, frame: np.ndarray) -> dict[str, Panel]:
        found = {panel.title: panel for panel in find_panels(frame, self.titles)}
        self.panels = found
        self.searched += 1
        return found

    def ensure(self, frame: np.ndarray, title: str) -> Panel | None:
        """要某個面板；手上沒有就重新搜尋一次。"""
        panel = self.panels.get(title)
        if panel is not None:
            return panel
        self.refresh(frame)
        return self.panels.get(title)

    def forget(self, title: str) -> None:
        """這個面板用不出東西了（被關掉或搬走），下次重新找。"""
        self.panels.pop(title, None)
