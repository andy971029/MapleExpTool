"""辨識「我是誰、在哪裡」—— 等級、職業、角色名、地圖。

這些欄位跟經驗值不一樣，不能全部用模板比對硬讀：

* **等級** 用的是另一套粗體字型，而且每個數字包在自己的橘色方塊裡。
  橘色很好認（色鍵一抓就中），所以定位沒問題；問題是手上只有「目前等級」
  那幾個數字的字形。解法是**邊玩邊學**：升級時我們從經驗歸零就知道等級 +1，
  於是新出現的字形可以自動標上正確的數字，久了十個數字就齊了。
* **職業與角色名** 是中文，而且角色名還可能有 ``卍`` 這種符號。這裡只負責定出
  那塊像素的範圍；文字由 Windows 內建 OCR 讀（``vision/ocr.py``），職業再跟清單
  比對（``core/jobvocab.py``）。像素的指紋用來判斷「有沒有變」—— 沒變就不重跑 OCR。
* **地圖** 要的是「區分地圖」而不是「讀出地圖名」，所以對那塊區域取雜湊就夠了：
  同一張地圖每次渲染的像素完全一樣，雜湊必定相同，所有統計都掛在這個雜湊上。
  名字另外用 OCR 讀、跟客戶端清單比對（``core/mapvocab.py``），比錯了改名也不會
  動到統計。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import ocr
from .preprocess import binarize, despeckle, to_luma
from .segment import column_runs, label_components, segment
from .templates import TemplateSet

# 等級方塊的橘色範圍（BGR）。遊戲的橘是高紅、中綠、低藍。
ORANGE_MIN_R = 170
ORANGE_G_RANGE = (70, 200)
ORANGE_MAX_B = 100

# 等級方塊的合理尺寸（像素）。
LEVEL_BOX_MIN_W = 8
LEVEL_BOX_MIN_H = 8
LEVEL_BOX_MAX_H = 48

# 方塊裡切出來的字形要像數字：尺寸在範圍內，而且每個數字高度要接近
# （實測誤判的例子切出 5x15 與 8x8 兩塊，一眼就知道不是同一行數字）。
LEVEL_DIGIT_MIN_W = 3
LEVEL_DIGIT_MAX_W = 18
LEVEL_DIGIT_MIN_H = 5
LEVEL_DIGIT_MAX_H = 22
LEVEL_DIGIT_HEIGHT_TOLERANCE = 2

# 角色名／職業那一塊從等級方塊右邊多遠開始找。
NAME_SEARCH_GAP = 18
# 寬度上限收得很緊：原本放到 420，只要狀態列上有別的亮東西（HP 標籤、
# 飄出來的訊息）就會被一路串連進來，浮窗上突然出現一大片截圖就是這樣來的。
NAME_MAX_WIDTH = 200
NAME_MAX_HEIGHT = 44
NAME_COLUMN_GAP = 14
NAME_MIN_INK = 0.03
NAME_MAX_INK = 0.55

# 小地圖面板的搜尋範圍（client 左上角的比例）。
MINIMAP_SEARCH_W = 0.35
MINIMAP_SEARCH_H = 0.30


def _crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = rect
    return frame[top:bottom, left:right]


def fingerprint(pixels: np.ndarray) -> str:
    """像素內容的短雜湊。同一張地圖/同一個角色 -> 完全相同的位元組 -> 相同雜湊。"""
    return hashlib.blake2s(
        np.ascontiguousarray(pixels[..., :3]).tobytes(), digest_size=6
    ).hexdigest()


# --------------------------------------------------------------------------- #
# 等級
# --------------------------------------------------------------------------- #


@dataclass
class LevelZone:
    rect: tuple[int, int, int, int]
    digits: list[np.ndarray] = field(default_factory=list, repr=False)
    image: np.ndarray | None = field(default=None, repr=False)
    """方塊的彩色原圖。模板認不出來時要拿它去走 OCR。"""

    @property
    def digit_count(self) -> int:
        return len(self.digits)

    @property
    def shape_key(self) -> str:
        """這組數字字形的指紋。等級變了，字形就變了。"""
        return "|".join(
            hashlib.blake2s(d.tobytes(), digest_size=4).hexdigest() for d in self.digits
        )


def find_orange_mask(frame: np.ndarray) -> np.ndarray:
    bgr = frame[..., :3].astype(np.int16)
    return (
        (bgr[..., 2] >= ORANGE_MIN_R)
        & (bgr[..., 1] >= ORANGE_G_RANGE[0])
        & (bgr[..., 1] <= ORANGE_G_RANGE[1])
        & (bgr[..., 0] <= ORANGE_MAX_B)
    )


def locate_level_zone(
    frame: np.ndarray, exp_rect: tuple[int, int, int, int] | None = None
) -> LevelZone | None:
    """找出等級的橘色數字方塊，並切出裡面的白色數字。

    一定要給 ``exp_rect``（已經定位好的經驗值欄位）當參考：畫面上橘色的東西
    很多（道具圖示、按鈕），光靠顏色會抓到背包裡的裝備。等級一定跟經驗值在
    **同一條狀態列、而且在它左邊**，這兩個條件一加就唯一了。
    """
    height, width = frame.shape[:2]
    if exp_rect is None:
        band_top = int(height * 0.8)
        band_bottom = height
        search_right = width
    else:
        centre = (exp_rect[1] + exp_rect[3]) // 2
        band_top = max(0, centre - 22)
        band_bottom = min(height, centre + 22)
        search_right = exp_rect[0]
    if band_bottom - band_top < 6 or search_right < 10:
        return None
    band = frame[band_top:band_bottom, :search_right]

    orange = despeckle(find_orange_mask(band))
    if not orange.any():
        return None

    # 相鄰的方塊（兩位數就是兩塊）合併成同一個等級區。
    runs = column_runs(orange)
    if not runs:
        return None
    groups: list[list[tuple[int, int]]] = []
    for run in runs:
        if groups and run[0] - groups[-1][-1][1] <= 6:
            groups[-1].append(run)
        else:
            groups.append([run])

    best: LevelZone | None = None
    for group in groups:
        left, right = group[0][0], group[-1][1]
        rows = np.flatnonzero(orange[:, left:right].any(axis=1))
        if rows.size == 0:
            continue
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        if right - left < LEVEL_BOX_MIN_W or not (
            LEVEL_BOX_MIN_H <= bottom - top <= LEVEL_BOX_MAX_H
        ):
            continue

        box = band[top:bottom, left:right]
        # 方塊裡的數字是白的，門檻拉高一點把橘底排掉。
        digits_mask = binarize(box, threshold=200, invert=False)
        glyphs = segment(digits_mask, min_ink=4, split_wide=False)
        if not _looks_like_digits([g.mask for g in glyphs]):
            continue
        zone = LevelZone(
            rect=(left, band_top + top, right, band_top + bottom),
            digits=[g.mask for g in glyphs],
            image=np.ascontiguousarray(box),
        )
        # 畫面上橘色的東西不只一個。同樣通過檢查時，取**離經驗值欄位最近**的
        # 那一組 —— 等級就在狀態列上，不會跑到畫面另一頭。
        if best is None or zone.rect[2] > best.rect[2]:
            best = zone
    return best


def _looks_like_digits(glyphs: list[np.ndarray]) -> bool:
    """切出來的東西像不像「一行等級數字」。"""
    if not glyphs or len(glyphs) > 3:
        return False
    heights = []
    for glyph in glyphs:
        height, width = glyph.shape
        if not (LEVEL_DIGIT_MIN_W <= width <= LEVEL_DIGIT_MAX_W):
            return False
        if not (LEVEL_DIGIT_MIN_H <= height <= LEVEL_DIGIT_MAX_H):
            return False
        heights.append(height)
    return max(heights) - min(heights) <= LEVEL_DIGIT_HEIGHT_TOLERANCE


# OCR 常把數字看成形狀相近的字母。等級一定是數字，所以直接換回去。
_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
     "l": "1", "I": "1", "i": "1", "|": "1", "!": "1",
     "Z": "2", "z": "2", "S": "5", "s": "5", "G": "6",
     "T": "7", "B": "8", "g": "9", "q": "9"}
)

# 等級方塊的調色盤：純白數字壓在橘底上，而且幾乎沒有抗鋸齒（實測 22x13 的方塊
# 只有 12 種顏色）。底色不寫死，由 coverage() 自己量 —— 橘底本身有深淺變化。
LEVEL_OCR_SCALES = (5, 6, 7, 8, 9, 10)

MAX_LEVEL = 300

# 同一組字形 OCR 連續失敗這麼多次就放棄它（字形變了會重新計算）。
OCR_RETRY_LIMIT = 3


def read_level_by_ocr(image: np.ndarray | None) -> int | None:
    """用 OCR 讀等級方塊裡的數字。

    為什麼需要這條路：模板比對是精確的，但得先**學過**那個字形才認得，而學習
    只發生在升級的當下。結果是玩家中途才開程式時，只要目前等級用到沒學過的
    數字就整個讀不出來（實測：模板檔裡只有「4」，所以 44 讀得到、45 就讀不到）。
    OCR 不用學。

    方塊只有 22x13，遠低於 OCR 引擎的設計尺寸，所以單一倍率不可靠 —— 實測掃過
    54 種（倍率 × 邊距）組合，只有 18 種會回答。但**回答的那些全部一致**，一個
    分歧都沒有。所以做法是多試幾個倍率，要全部同意才採用；有任何一個不一樣就
    當作讀不到，寧可留空。
    """
    if image is None or image.size == 0:
        return None
    answers = set()
    for scale in LEVEL_OCR_SCALES:
        text = ocr.recognize(image, scale=scale)
        if not text:
            continue
        cleaned = text.translate(_DIGIT_LOOKALIKES).strip()
        if not cleaned.isdigit() or not 1 <= len(cleaned) <= 3:
            return None          # 讀出不像數字的東西 -> 這次整個不信
        value = int(cleaned)
        if not 1 <= value <= MAX_LEVEL:
            return None
        answers.add(value)
    if len(answers) != 1:
        return None              # 沒有人回答、或有人不同意
    return answers.pop()


class LevelReader:
    """讀等級。先用字型模板，認不出來才退到 OCR。

    兩條路各有長短，所以兩條都留：

    * **模板比對**精確又快（整塊方塊 1.3ms），但要先學過那個字形。學習發生在
      升級的當下（新等級必定是舊等級 +1），所以玩久了十個數字會自己補齊。
    * **OCR** 不用學，但方塊只有 22x13，得靠「多個倍率互相印證」才敢信。

    OCR 讀出來之後會順手把字形**教給模板**，下次就走快路 —— 等於用 OCR 自動
    完成原本要玩家跑 ``level`` 指令才做得到的初始化。
    """

    def __init__(self, templates: TemplateSet | None = None) -> None:
        self.templates = templates or TemplateSet()
        self.level: int | None = None
        self.learned = 0
        # 同一組字形 OCR 失敗幾次就別再試了。OCR 這條路要 66ms，而字形沒變的話
        # 重試的結果也不會變 —— 不擋的話會變成每秒白花 66ms。
        self._ocr_misses: dict[str, int] = {}

    def read(self, zone: LevelZone | None) -> int | None:
        if zone is None or not zone.digits:
            return None
        value = self.read_templates(zone)
        if value is not None:
            self.level = value
            return value

        key = zone.shape_key
        if self._ocr_misses.get(key, 0) >= OCR_RETRY_LIMIT:
            return None
        value = read_level_by_ocr(zone.image)
        if value is None:
            self._ocr_misses[key] = self._ocr_misses.get(key, 0) + 1
            return None
        self._ocr_misses.pop(key, None)
        # 字數對得上才教 —— 對不上代表切字形跟 OCR 看到的不是同一回事，
        # 教下去會把錯的字形永久記起來。
        if len(str(value)) == len(zone.digits):
            self.teach(zone, value)
        self.level = value
        return value

    def read_templates(self, zone: LevelZone) -> int | None:
        chars = []
        for digit in zone.digits:
            match = self.templates.match(digit, min_score=0.85, min_margin=0.03)
            if not match.char or not match.char.isdigit():
                return None
            chars.append(match.char)
        try:
            value = int("".join(chars))
        except ValueError:
            return None
        return value if 1 <= value <= MAX_LEVEL else None

    def teach(self, zone: LevelZone | None, level: int) -> int:
        """已知等級時，把這組字形學起來。回傳新學會的字形數。"""
        if zone is None or not zone.digits:
            return 0
        text = str(level)
        if len(text) != len(zone.digits):
            return 0
        added = 0
        for char, digit in zip(text, zone.digits):
            if self.templates.add(char, digit):
                added += 1
        self.learned += added
        self.level = level
        return added

    @property
    def known_digits(self) -> str:
        return "".join(sorted(set("".join(self.templates.chars()))))


# --------------------------------------------------------------------------- #
# 角色名與職業
# --------------------------------------------------------------------------- #


def locate_name_zone(
    frame: np.ndarray, level_rect: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    """等級方塊右邊那塊（職業 + 角色名）。

    不去讀字，只定出範圍 —— 那塊像素會原樣顯示在浮窗上。
    """
    if level_rect is None:
        return None
    height, width = frame.shape[:2]
    start_x = min(width - 1, level_rect[2] + NAME_SEARCH_GAP)
    top = max(0, level_rect[1] - 12)
    bottom = min(height, level_rect[3] + 12)
    end_x = min(width, start_x + NAME_MAX_WIDTH)
    if end_x - start_x < 10:
        return None

    strip = frame[top:bottom, start_x:end_x]
    mask = despeckle(binarize(strip, threshold=170, invert=False))
    runs = column_runs(mask)
    if not runs:
        return None

    # 從左邊第一塊文字開始，遇到大空隙就停（那是跟 HP 欄位之間的間隔）。
    left = runs[0][0]
    right = runs[0][1]
    for previous, current in zip(runs, runs[1:]):
        if current[0] - previous[1] > NAME_COLUMN_GAP:
            break
        right = current[1]

    rows = np.flatnonzero(mask[:, left:right].any(axis=1))
    if rows.size == 0:
        return None
    rect = (
        start_x + left - 2,
        top + int(rows[0]) - 2,
        start_x + right + 2,
        top + int(rows[-1]) + 3,
    )
    # 最後再驗一次尺寸與墨水比例。失敗就回 None（上層會沿用上一張好的），
    # 總比在浮窗上貼一塊莫名其妙的截圖好。
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    if not (10 <= width <= NAME_MAX_WIDTH and 6 <= height <= NAME_MAX_HEIGHT):
        return None
    ink = float(mask[:, left:right].mean())
    if not (NAME_MIN_INK <= ink <= NAME_MAX_INK):
        return None
    return rect


# --------------------------------------------------------------------------- #
# 地圖
# --------------------------------------------------------------------------- #


# 小地圖「名稱區」的底色：一種偏藍的淺鋼藍，而且由上而下有漸層。
# 判準除了數值範圍，還要求 B >= G >= R 且藍紅差夠大 —— 遊戲場景很少同時滿足。
PANEL_B = (190, 242)
PANEL_G = (172, 222)
PANEL_R = (138, 204)
PANEL_MIN_BLUE_BIAS = 25

# 名稱區的合理高度，以及判定「這一列屬於名稱區」的面板色像素門檻。
PANEL_BAND_MIN_H = 20
PANEL_BAND_MAX_H = 80
# 判定「這一列屬於名稱區」的門檻，相對於面板寬度。
# 不能用「相對於畫面上最大面板」的比例（角色資料視窗、背包面積更大，會把門檻
# 拉高），也不能訂得太高：長地圖名（例如「廢棄的妖精的圖書館」9 個字）會把
# 背景吃掉大半，實測侵蝕後每列只剩 12–14，而真正的空白列是 0。
PANEL_ROW_DIVISOR = 28
PANEL_ROW_MIN_PIXELS = 6
# 容忍帶內幾列低於門檻 —— 字特別密的那幾列不該把整條帶切斷。
PANEL_ROW_GAP_TOLERANCE = 3
PANEL_COL_RATIO = 0.30


# 名稱區裡，圖示與文字之間的空隙。
MAP_TEXT_GAP = 4
# 圖示的特徵：接近正方形、而且幾乎佔滿整條名稱區的高度。
ICON_MAX_ASPECT_DIFF = 0.35
ICON_MIN_HEIGHT_RATIO = 0.75

# 地圖要連續這麼多次讀到相同指紋才算換圖。
# 這是真正的把關：UI 是靜態的，遊戲畫面每一幀都在動，動的東西穩定不下來。
MAP_STABLE_SAMPLES = 3


def panel_mask(frame: np.ndarray) -> np.ndarray:
    bgr = frame[..., :3].astype(np.int16)
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return (
        (blue >= PANEL_B[0])
        & (blue <= PANEL_B[1])
        & (green >= PANEL_G[0])
        & (green <= PANEL_G[1])
        & (red >= PANEL_R[0])
        & (red <= PANEL_R[1])
        & (blue >= green)
        & (green >= red)
        & ((blue - red) >= PANEL_MIN_BLUE_BIAS)
    )


def locate_map_zone(
    frame: np.ndarray, search_rect: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int] | None:
    """小地圖面板上的地圖名稱那一塊。

    ``search_rect`` 應該由 :mod:`mapleexp.vision.panels` 先定位出「小地圖」面板
    再傳進來。單靠顏色是分不開的：角色資料視窗、背包面板用的是**同一種淺鋼藍**
    （實測會抓到背包裡的「124,523 楓幣」）。曾經改用「只找畫面左上角」當權宜，
    但玩家把小地圖拖走之後就整個失效 —— 面板位置本來就不該假設。
    """
    height, width = frame.shape[:2]
    if search_rect is not None:
        left, top, right, bottom = search_rect
    else:
        left, top, right, bottom = 0, 0, width, height
    region = frame[top:bottom, left:right]
    offset = (left, top)
    if region.size == 0:
        return None

    mask = panel_mask(region)
    if not mask.any():
        return None

    for band_top, band_bottom in _candidate_bands(mask):
        columns = mask[band_top:band_bottom, :].sum(axis=0)
        hot = columns >= (band_bottom - band_top) * PANEL_COL_RATIO
        # 同一條橫列上可能同時有小地圖與角色資料視窗（兩者底色相同）。
        # 欄位也要分組，取**最左邊**那一塊 —— 小地圖在左上角。
        blocks = [
            (a, b) for a, b in _column_groups(column_runs(hot[None, :]), 8)
            if b - a >= 30
        ]
        if not blocks:
            continue
        band_left, band_right = blocks[0]

        inner = region[band_top:band_bottom, band_left:band_right]
        text = despeckle(binarize(inner, threshold=120, invert=True))
        if not text.any():
            continue

        groups = _column_groups(column_runs(text), MAP_TEXT_GAP)
        kept = [
            g for g in groups
            if not _looks_like_icon(text, g, band_bottom - band_top)
        ]
        if not kept:
            continue

        inner_left = min(g[0] for g in kept)
        inner_right = max(g[1] for g in kept)
        rows = np.flatnonzero(text[:, inner_left:inner_right].any(axis=1))
        if rows.size == 0:
            continue

        return (
            offset[0] + band_left + inner_left - 1,
            offset[1] + band_top + int(rows[0]) - 1,
            min(width, offset[0] + band_left + inner_right + 1),
            min(height, offset[1] + band_top + int(rows[-1]) + 2),
        )
    return None


def _candidate_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """所有可能是「面板名稱區」的橫帶，由上而下排序。

    先侵蝕一次：小地圖**標題列**的外框也是同一種藍，而且它的列密度剛好跟
    「文字很密的那幾列」重疊，不侵蝕的話會把標題列跟名稱區黏成一條。
    框線只有一兩個像素寬，侵蝕一次就不見了，名稱區的實心底色則留得下來。
    """
    eroded = _erode(mask, 1)
    if not eroded.any():
        return []
    counts = eroded.sum(axis=1)
    threshold = max(PANEL_ROW_MIN_PIXELS, eroded.shape[1] // PANEL_ROW_DIVISOR)

    bands: list[tuple[int, int]] = []
    start: int | None = None
    last_hot: int | None = None
    for y in range(eroded.shape[0] + 1):
        hot = y < eroded.shape[0] and counts[y] >= threshold
        if hot:
            if start is None:
                start = y
            last_hot = y
            continue
        if start is None:
            continue
        # 容忍幾列低於門檻；超過才算這條帶結束。
        if last_hot is not None and y - last_hot <= PANEL_ROW_GAP_TOLERANCE:
            continue
        end = (last_hot or start) + 1
        if PANEL_BAND_MIN_H <= end - start <= PANEL_BAND_MAX_H:
            bands.append((start, end))
        start = None
        last_hot = None
    return bands


def _erode(mask: np.ndarray, times: int = 2) -> np.ndarray:
    """四鄰侵蝕。用來把細的框線去掉，只留下實心色塊。"""
    result = mask
    for _ in range(max(0, times)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        height, width = result.shape
        result = (
            padded[0:height, 1 : width + 1]
            & padded[2 : height + 2, 1 : width + 1]
            & padded[1 : height + 1, 0:width]
            & padded[1 : height + 1, 2 : width + 2]
            & result
        )
    return result


def _column_groups(
    runs: list[tuple[int, int]], max_gap: int
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    for run in runs:
        if groups and run[0] - groups[-1][1] <= max_gap:
            groups[-1] = (groups[-1][0], run[1])
        else:
            groups.append(run)
    return groups


def _looks_like_icon(
    text: np.ndarray, group: tuple[int, int], band_height: int
) -> bool:
    """左邊那個地圖圖示：接近正方形，而且幾乎佔滿整條名稱區的高度。"""
    left, right = group
    rows = np.flatnonzero(text[:, left:right].any(axis=1))
    if rows.size == 0:
        return True
    group_width = right - left
    group_height = int(rows[-1] - rows[0]) + 1
    if group_height < band_height * ICON_MIN_HEIGHT_RATIO:
        return False
    if group_width <= 0:
        return True
    aspect = abs(group_width - group_height) / max(group_width, group_height)
    return aspect <= ICON_MAX_ASPECT_DIFF


class MapWatcher:
    """追蹤目前在哪張地圖。

    只在指紋連續穩定數次之後才認定換圖。這一條就擋掉了所有誤判：
    小地圖被收合、被移動、或預設位置剛好對到遊戲場景時，那塊區域每一幀都在變，
    永遠穩定不下來，於是地圖維持「未知」而不是亂跳。
    """

    def __init__(self, stable_samples: int = MAP_STABLE_SAMPLES) -> None:
        self.stable_samples = max(1, stable_samples)
        self.map_id = ""
        self.image: np.ndarray | None = None
        self._candidate = ""
        self._candidate_image: np.ndarray | None = None
        self._streak = 0

    def feed(self, map_id: str, image: np.ndarray | None) -> bool:
        """餵入一次觀測。回傳是否發生了換圖。"""
        if not map_id:
            self._candidate = ""
            self._streak = 0
            return False
        if map_id == self._candidate:
            self._streak += 1
        else:
            self._candidate = map_id
            self._candidate_image = image
            self._streak = 1

        if self._streak >= self.stable_samples and map_id != self.map_id:
            self.map_id = map_id
            self.image = self._candidate_image
            return True
        return False


def text_bands(image: np.ndarray, min_height: int = 5) -> list[tuple[int, int]]:
    """把一塊畫面依「有沒有亮像素」切成一行一行。"""
    luma = to_luma(image)
    rows = luma >= max(120, int(luma.mean()) + 20)
    occupied = rows.any(axis=1)
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(len(occupied) + 1):
        hot = y < len(occupied) and bool(occupied[y])
        if hot and start is None:
            start = y
        elif not hot and start is not None:
            if y - start >= min_height:
                bands.append((max(0, start - 1), min(len(occupied), y + 1)))
            start = None
    return bands


@dataclass
class Identity:
    """一次掃描的結果。``*_image`` 是給浮窗直接畫出來的像素。"""

    level: int | None = None
    level_rect: tuple[int, int, int, int] | None = None
    level_shape_key: str = ""
    # 等級字形區留著，升級時要靠它把新數字學起來。
    zone: "LevelZone | None" = field(default=None, repr=False)
    name_rect: tuple[int, int, int, int] | None = None
    name_image: np.ndarray | None = field(default=None, repr=False)
    map_rect: tuple[int, int, int, int] | None = None
    map_image: np.ndarray | None = field(default=None, repr=False)
    map_id: str = ""


def scan(
    frame: np.ndarray,
    level_reader: LevelReader | None = None,
    exp_rect: tuple[int, int, int, int] | None = None,
    map_rect: tuple[int, int, int, int] | None = None,
) -> Identity:
    """掃一次畫面，取得等級／角色名／地圖。"""
    result = Identity()

    zone = locate_level_zone(frame, exp_rect)
    if zone is not None:
        result.zone = zone
        result.level_rect = zone.rect
        result.level_shape_key = zone.shape_key
        if level_reader is not None:
            result.level = level_reader.read(zone)

    name_rect = locate_name_zone(frame, result.level_rect)
    if name_rect is not None:
        result.name_rect = name_rect
        result.name_image = _crop(frame, name_rect).copy()

    # 地圖一定要先有面板範圍才找 —— 沒有範圍時整張畫面掃，必定誤判。
    found_map = locate_map_zone(frame, map_rect) if map_rect is not None else None
    if found_map is not None:
        result.map_rect = found_map
        crop = _crop(frame, found_map)
        result.map_image = crop.copy()
        result.map_id = fingerprint(crop)

    return result
