"""把 ROI 讀成一筆 :class:`StatusReading`。

這一層只負責「畫面 -> 字串 -> 數值」，不做任何統計判斷。
失敗時一律回傳 ``ok=False`` 加上具體原因，讓追蹤器自己決定要暫停還是忽略 ——
猜一個數字遠比承認這一格讀不到更糟。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ReaderConfig, Roi
from ..core.reading import StatusReading, parse_exp_text
from ..win32.capture import CaptureError, WindowCapturer
from . import pngio
from .preprocess import binarize
from .segment import GlyphBox, segment
from .templates import TemplateSet


@dataclass
class RoiRead:
    """單一 ROI 的辨識結果（含除錯用的中間產物）。"""

    text: str
    scores: list[float] = field(default_factory=list)
    boxes: list[GlyphBox] = field(default_factory=list)
    unresolved: int = 0
    mask: np.ndarray | None = field(default=None, repr=False)
    pixels: np.ndarray | None = field(default=None, repr=False)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text) and self.unresolved == 0

    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 0.0


class StatusReader:
    """用擷取器 + 模板，把狀態列讀成數值。"""

    def __init__(
        self,
        capturer: WindowCapturer,
        config: ReaderConfig,
        templates: TemplateSet,
        debug_dir: Path | None = None,
    ) -> None:
        self.capturer = capturer
        self.config = config
        self.templates = templates
        self.debug_dir = debug_dir
        self._debug_saved = 0

    # ------------------------------------------------------------------ #

    def read_roi(self, roi: Roi, keep_images: bool = False) -> RoiRead:
        if not roi.is_set():
            return RoiRead(text="", error="ROI 尚未校準")

        try:
            frame = self.capturer.capture_roi(roi)
        except CaptureError as exc:
            return RoiRead(text="", error=str(exc))

        cfg = self.config
        mask = binarize(
            frame.pixels,
            threshold=cfg.threshold,
            invert=cfg.invert,
            ink_color=cfg.ink_color,
            color_tolerance=cfg.color_tolerance,
        )
        boxes = segment(
            mask,
            min_width=cfg.min_glyph_width,
            min_height=cfg.min_glyph_height,
            split_wide=cfg.split_wide_glyphs,
            expected_width=self.templates.median_width(),
        )

        result = RoiRead(
            text="",
            boxes=boxes,
            mask=mask if keep_images else None,
            pixels=frame.pixels if keep_images else None,
        )
        if not boxes:
            result.error = "ROI 內找不到任何文字（門檻值或 ROI 位置可能不對）"
            return result

        characters: list[str] = []
        for box in boxes:
            match = self.templates.match(
                box.extract(mask),
                min_score=cfg.match_min_score,
                min_margin=cfg.match_min_margin,
            )
            result.scores.append(match.score)
            if match.char:
                characters.append(match.char)
            else:
                result.unresolved += 1
                characters.append("?")

        result.text = "".join(characters)
        if result.unresolved:
            result.error = (
                f"{result.unresolved} 個字元無法辨識（最低分 {result.min_score:.2f}）"
            )
        return result

    # ------------------------------------------------------------------ #

    def read(self) -> StatusReading:
        mono = time.perf_counter()
        wall = time.time()

        exp_read = self.read_roi(self.config.exp_roi, keep_images=self.debug_dir is not None)
        if not exp_read.ok:
            self._maybe_dump(exp_read)
            return StatusReading.failed(
                mono, wall, reason=exp_read.error or "經驗值辨識失敗", raw_exp=exp_read.text
            )

        exp_abs, exp_pct = parse_exp_text(exp_read.text)
        if exp_abs is None:
            self._maybe_dump(exp_read)
            return StatusReading.failed(
                mono,
                wall,
                reason=f"無法從 {exp_read.text!r} 取出絕對經驗值",
                raw_exp=exp_read.text,
            )

        # 等級不在這裡讀：它從狀態列的橘色方塊自動定位（vision/identity.py），
        # 由 TrackingSession 在餵給追蹤器之前補上。
        return StatusReading(
            mono=mono,
            wall=wall,
            ok=True,
            exp_abs=exp_abs,
            exp_pct=exp_pct,
            raw_exp=exp_read.text,
            scores=tuple(exp_read.scores),
        )

    # ------------------------------------------------------------------ #

    def _maybe_dump(self, read: RoiRead, limit: int = 40) -> None:
        """把辨識失敗的畫面存成 PNG，方便事後查為什麼讀不到。"""
        if self.debug_dir is None or self._debug_saved >= limit:
            return
        if read.pixels is None and read.mask is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{self._debug_saved:03d}"
        try:
            if read.pixels is not None:
                (self.debug_dir / f"{stamp}-roi.png").write_bytes(
                    pngio.encode_png(
                        pngio.scale_nearest(pngio.bgra_to_rgb(read.pixels), 3)
                    )
                )
            if read.mask is not None:
                (self.debug_dir / f"{stamp}-mask.png").write_bytes(
                    pngio.encode_png(pngio.scale_nearest(pngio.mask_to_rgb(read.mask), 3))
                )
        except OSError:
            return
        self._debug_saved += 1
