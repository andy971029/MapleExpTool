"""字元模板比對。

為什麼用模板比對而不是 OCR：狀態列的經驗值是固定 UI 字型、固定大小、固定顏色，
要辨識的字形只有 ``0-9`` 加上幾個符號。校準時各存一張模板，之後逐格比對，
準確率實質上是 100%，單次不到 1 毫秒，而且不用背一個 OCR 引擎的相依。
通用 OCR 反而容易在小字號上把 ``8`` 讀成 ``B``、``1`` 讀成 ``l``。

相似度用 IoU（交集/聯集）而不是單純的「相同像素比例」：
後者會因為背景像素佔多數而讓所有分數都很高、彼此難以區分；
IoU 同時懲罰「少畫」與「多畫」，對字形差異敏感得多。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .segment import normalize_glyph

# 比對時允許的平移量（像素）。抗鋸齒與子像素定位會讓同一個字元位移 ±1。
MAX_SHIFT = 1

# 尺寸差超過這個值就不比了（省時間，也避免荒謬的配對）。
SIZE_PREFILTER = 3


@dataclass(frozen=True)
class MatchResult:
    char: str
    score: float
    margin: float
    runner_up: str = ""

    @property
    def confident(self) -> bool:
        return bool(self.char)


@dataclass
class Template:
    char: str
    mask: np.ndarray = field(repr=False)

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])


def iou(a: np.ndarray, b: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    """兩個遮罩在允許小幅平移下的最佳 IoU。"""
    if a.size == 0 or b.size == 0:
        return 0.0
    height = max(a.shape[0], b.shape[0]) + 2 * max_shift
    width = max(a.shape[1], b.shape[1]) + 2 * max_shift

    canvas_b = np.zeros((height, width), dtype=bool)
    canvas_b[max_shift : max_shift + b.shape[0], max_shift : max_shift + b.shape[1]] = b

    best = 0.0
    canvas_a = np.zeros((height, width), dtype=bool)
    for dy in range(2 * max_shift + 1):
        for dx in range(2 * max_shift + 1):
            canvas_a[:] = False
            canvas_a[dy : dy + a.shape[0], dx : dx + a.shape[1]] = a
            intersection = int(np.count_nonzero(canvas_a & canvas_b))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(canvas_a | canvas_b))
            if union:
                score = intersection / union
                if score > best:
                    best = score
    return best


class TemplateSet:
    """一組字元模板。同一個字元可以有多張樣本（不同位移、不同畫面狀態）。"""

    def __init__(self) -> None:
        self._templates: list[Template] = []
        self.meta: dict = {}

    # ------------------------------------------------------------------ #
    # 建立
    # ------------------------------------------------------------------ #

    def add(self, char: str, mask: np.ndarray) -> bool:
        """加入一張模板。回傳是否真的加入（重複的樣本會被忽略）。

        ``char`` 可以是多個字元。這是為了處理「兩個字形靠對角線貼在一起、
        切不開」的情況（實測到的例子是 ``[28.37%]`` 的小數點碰到 3 的左下角）：
        直接把那個形狀標成 ``.3``，比起勉強去切開它穩定得多。
        """
        if not char:
            return False
        trimmed = normalize_glyph(np.asarray(mask, dtype=bool))
        if trimmed.size == 0:
            return False
        for existing in self._templates:
            if existing.char == char and existing.mask.shape == trimmed.shape:
                if np.array_equal(existing.mask, trimmed):
                    return False
        self._templates.append(Template(char=char, mask=trimmed))
        return True

    def remove_char(self, char: str) -> int:
        before = len(self._templates)
        self._templates = [t for t in self._templates if t.char != char]
        return before - len(self._templates)

    def clear(self) -> None:
        self._templates.clear()

    # ------------------------------------------------------------------ #
    # 查詢
    # ------------------------------------------------------------------ #

    def chars(self) -> list[str]:
        return sorted({t.char for t in self._templates})

    def count_for(self, char: str) -> int:
        return sum(1 for t in self._templates if t.char == char)

    def masks_for(self, char: str) -> list[np.ndarray]:
        """某個字元的所有模板樣本。自動定位時拿它當錨點比對。"""
        return [t.mask for t in self._templates if t.char == char]

    def median_width(self, chars: str = "0123456789") -> float | None:
        """數字模板寬度的中位數，給切割層當「一個字元該有多寬」的基準。"""
        widths = [t.width for t in self._templates if t.char in chars]
        if not widths:
            return None
        return float(np.median(widths))

    def __len__(self) -> int:
        return len(self._templates)

    def match(
        self,
        mask: np.ndarray,
        min_score: float = 0.80,
        min_margin: float = 0.02,
    ) -> MatchResult:
        """比對單一字元。

        分數不夠高、或前兩名差距太小（例如 ``8`` 與 ``9`` 難分）時，
        回傳空字元讓上層判定為辨識失敗 —— 寧可這一格丟掉，也不要記下錯的數字。
        """
        candidate = normalize_glyph(np.asarray(mask, dtype=bool))
        if candidate.size == 0 or not self._templates:
            return MatchResult(char="", score=0.0, margin=0.0)

        best_by_char: dict[str, float] = {}
        for template in self._templates:
            if (
                abs(template.height - candidate.shape[0]) > SIZE_PREFILTER
                or abs(template.width - candidate.shape[1]) > SIZE_PREFILTER
            ):
                continue
            score = iou(candidate, template.mask)
            if score > best_by_char.get(template.char, 0.0):
                best_by_char[template.char] = score

        if not best_by_char:
            return MatchResult(char="", score=0.0, margin=0.0)

        ranked = sorted(best_by_char.items(), key=lambda item: item[1], reverse=True)
        top_char, top_score = ranked[0]
        second_char, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        margin = top_score - second_score

        if top_score < min_score or margin < min_margin:
            return MatchResult(
                char="", score=top_score, margin=margin, runner_up=second_char
            )
        return MatchResult(
            char=top_char, score=top_score, margin=margin, runner_up=second_char
        )

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        entries = []
        for template in self._templates:
            flat = template.mask.reshape(-1)
            entries.append(
                {
                    "char": template.char,
                    "h": template.height,
                    "w": template.width,
                    "bits": base64.b64encode(np.packbits(flat).tobytes()).decode("ascii"),
                    # 純粹為了讓人（和 diff）看得懂，載入時不使用。
                    "preview": preview_lines(template.mask),
                }
            )
        return {"version": 1, "meta": dict(self.meta), "templates": entries}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(path)
        return path

    @classmethod
    def from_dict(cls, raw: dict) -> "TemplateSet":
        instance = cls()
        if not isinstance(raw, dict):
            return instance
        meta = raw.get("meta")
        if isinstance(meta, dict):
            instance.meta = dict(meta)
        for entry in raw.get("templates") or []:
            if not isinstance(entry, dict):
                continue
            try:
                char = str(entry["char"])
                height = int(entry["h"])
                width = int(entry["w"])
                packed = np.frombuffer(base64.b64decode(entry["bits"]), dtype=np.uint8)
            except (KeyError, TypeError, ValueError):
                continue
            if height <= 0 or width <= 0:
                continue
            bits = np.unpackbits(packed)
            if bits.size < height * width:
                continue
            mask = bits[: height * width].reshape(height, width).astype(bool)
            instance.add(char, mask)
        return instance

    @classmethod
    def load(cls, path: Path) -> "TemplateSet":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls.from_dict(raw)


def preview_lines(mask: np.ndarray) -> list[str]:
    """把遮罩畫成 ASCII，方便在終端機或 JSON 裡用眼睛檢查模板對不對。"""
    return ["".join("#" if value else "." for value in row) for row in mask]
