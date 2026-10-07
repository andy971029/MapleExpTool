"""把一條狀態列的 ink 遮罩切成一個一個字元。

切割回傳的是**每個字元自己的像素**，而不是一塊矩形裁切。這個差別很重要：

``[27.50%]`` 裡的小數點與後面的數字其實是兩個互不相連的區塊，但兩者的外接矩形
在欄位上重疊了一格。矩形裁切沒辦法把它們分開（任何一條垂直切線都會切掉
其中一邊的像素），只有「依連通區塊取出各自的像素」才分得乾淨。

分組規則：兩個連通區塊屬於同一個字元，條件是它們的欄位範圍**重疊超過一格**，
或其中一個完全包在另一個裡面。這樣 ``%`` 那種由左圈、斜線、右圈組成的字元
會被正確地留在一起（斜線橫跨整個寬度、包住另外兩塊），而只碰到一格的
小數點則會被分出去。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocess import despeckle

# 寬度超過基準這個倍率的區塊，視為多個字元黏在一起。
WIDE_SPLIT_RATIO = 1.8

# 欄位重疊超過這麼多格才算同一個字元。
SAME_GLYPH_MIN_OVERLAP = 2


@dataclass(frozen=True)
class Glyph:
    """切出來的一個字元。

    ``mask`` 已經是這個字元自己的像素（鄰居的像素不會混進來），裁到自己的外接矩形。
    """

    left: int
    top: int
    right: int
    bottom: int
    mask: np.ndarray = field(repr=False)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def ink(self) -> int:
        return int(np.count_nonzero(self.mask))

    def extract(self, _mask: np.ndarray | None = None) -> np.ndarray:
        """取得字元遮罩。

        參數留著是為了相容舊的呼叫方式；切割時就已經把像素取出來了，
        所以這裡不需要再從整條遮罩去裁。
        """
        return self.mask


# 舊名稱，保留相容。
GlyphBox = Glyph


def column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """有 ink 的連續欄位區間 [(start, end), ...]，end 為開區間。"""
    if mask.size == 0:
        return []
    occupied = mask.any(axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(occupied):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(occupied)))
    return runs


def label_components(
    mask: np.ndarray, connectivity: int = 8
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    """連通區塊標記。

    回傳 (labels, boxes)：``labels`` 與 mask 同形狀、背景為 0；
    ``boxes`` 是 ``{label: (left, top, right, bottom)}``。

    遮罩只有狀態列那一小條（約 15x80），用純 Python 跑 union-find 完全夠快，
    省掉一個影像處理相依。
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    offsets = (
        ((-1, -1), (-1, 0), (-1, 1), (0, -1))
        if connectivity == 8
        else ((-1, 0), (0, -1))
    )

    for y in range(height):
        row = mask[y]
        for x in range(width):
            if not row[x]:
                continue
            neighbours = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and labels[ny, nx]:
                    neighbours.append(int(labels[ny, nx]))
            if not neighbours:
                parent.append(len(parent))
                labels[y, x] = len(parent) - 1
            else:
                smallest = min(neighbours)
                labels[y, x] = smallest
                for other in neighbours:
                    union(smallest, other)

    # 把等價的標籤收斂到同一個根。
    if len(parent) > 1:
        lookup = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
        labels = lookup[labels]

    boxes: dict[int, list[int]] = {}
    ys, xs = np.nonzero(labels)
    for y, x in zip(ys.tolist(), xs.tolist()):
        label = int(labels[y, x])
        box = boxes.get(label)
        if box is None:
            boxes[label] = [x, y, x + 1, y + 1]
        else:
            box[0] = min(box[0], x)
            box[1] = min(box[1], y)
            box[2] = max(box[2], x + 1)
            box[3] = max(box[3], y + 1)
    return labels, {k: tuple(v) for k, v in boxes.items()}


def connected_components(
    mask: np.ndarray, connectivity: int = 8
) -> list[tuple[int, int, int, int]]:
    """連通區塊的外接矩形清單，由左到右。"""
    _labels, boxes = label_components(mask, connectivity)
    return sorted(boxes.values(), key=lambda b: b[0])


def _same_glyph(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """兩個連通區塊是不是同一個字元的部件。"""
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap >= SAME_GLYPH_MIN_OVERLAP:
        return True
    # 一方完全包住另一方（``%`` 的斜線橫跨整個寬度就是這種情況）。
    return (a[0] <= b[0] and a[2] >= b[2]) or (b[0] <= a[0] and b[2] >= a[2])


def _group_components(
    boxes: dict[int, tuple[int, int, int, int]],
) -> list[list[int]]:
    """把連通區塊分組，每一組就是一個字元。由左至右排序。"""
    labels = sorted(boxes, key=lambda label: boxes[label][0])
    groups: list[list[int]] = []
    for label in labels:
        box = boxes[label]
        for group in groups:
            if any(_same_glyph(box, boxes[other]) for other in group):
                group.append(label)
                break
        else:
            groups.append([label])
    groups.sort(key=lambda group: min(boxes[label][0] for label in group))
    return groups


def segment(
    mask: np.ndarray,
    min_width: int = 2,
    min_height: int = 3,
    split_wide: bool = True,
    expected_width: float | None = None,
    min_ink: int = 3,
    remove_specks: bool = True,
) -> list[Glyph]:
    """切出字元，由左至右排序。

    ``expected_width`` 是單一字元的預期寬度。有模板時應該傳模板寬度的中位數 ——
    那比從當前畫面推估可靠得多，而且在「所有字元都黏成一塊」時仍然有效。
    """
    if remove_specks:
        mask = despeckle(mask)
    if mask.size == 0 or not mask.any():
        return []

    glyphs: list[Glyph] = []
    for run_left, run_right in column_runs(mask):
        sub = mask[:, run_left:run_right]
        labels, boxes = label_components(sub, connectivity=8)
        if not boxes:
            continue

        for group in _group_components(boxes):
            group_mask = np.isin(labels, group)
            left = min(boxes[label][0] for label in group)
            right = max(boxes[label][2] for label in group)
            for piece_left, piece_right in _split_wide(
                group_mask, left, right, split_wide, expected_width
            ):
                piece = group_mask[:, piece_left:piece_right]
                rows = np.flatnonzero(piece.any(axis=1))
                if rows.size == 0:
                    continue
                top, bottom = int(rows[0]), int(rows[-1]) + 1
                trimmed = piece[top:bottom]
                glyph = Glyph(
                    left=run_left + piece_left,
                    top=top,
                    right=run_left + piece_right,
                    bottom=bottom,
                    mask=trimmed,
                )
                # 只有「兩個維度都太小」才算雜點。小數點本身就只有 2x2，
                # 但它是解析百分比的關鍵字元，絕對不能被濾掉。
                if glyph.width < min_width and glyph.height < min_height:
                    continue
                if glyph.ink < min_ink:
                    continue
                glyphs.append(glyph)

    glyphs.sort(key=lambda g: g.left)
    return glyphs


def _split_wide(
    group_mask: np.ndarray,
    left: int,
    right: int,
    enabled: bool,
    expected_width: float | None,
) -> list[tuple[int, int]]:
    """把明顯過寬的區塊等分切開（字元真的黏在一起時）。

    等分是合理的做法，因為數字在 UI 字型裡是等寬的；真正的寬字元
    （``%`` 之類）只有一個，不會把基準寬度拉高到誤判的程度。
    """
    width = right - left
    if not enabled or not expected_width or expected_width <= 0:
        return [(left, right)]
    if width < expected_width * WIDE_SPLIT_RATIO:
        return [(left, right)]
    parts = int(round(width / expected_width))
    if parts < 2:
        return [(left, right)]
    edges = np.linspace(left, right, parts + 1).round().astype(int)
    return [
        (int(a), int(b)) for a, b in zip(edges, edges[1:]) if int(b) > int(a)
    ]


def normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    """把字元遮罩裁到外接矩形，作為模板比對的標準形式。"""
    if glyph.size == 0 or not glyph.any():
        return glyph[:0, :0]
    rows = np.flatnonzero(glyph.any(axis=1))
    cols = np.flatnonzero(glyph.any(axis=0))
    return glyph[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
