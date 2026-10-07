"""內建的 5x7 點陣字型，只用於自我測試。

真正的模板是校準時從遊戲畫面取下來的；這組字型存在的理由是讓
「二值化 -> 切割 -> 模板比對 -> 解析」整條管線可以在沒有遊戲、沒有截圖的
情況下被完整測試。管線本身出問題和校準沒做好是兩件事，要能分開驗證。
"""

from __future__ import annotations

import numpy as np

GLYPHS: dict[str, list[str]] = {
    "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    "%": ["11001", "11010", "00100", "00100", "00100", "01011", "10011"],
}

GLYPH_HEIGHT = 7
GLYPH_WIDTH = 5


def glyph_mask(char: str) -> np.ndarray:
    """單一字元的遮罩。"""
    rows = GLYPHS.get(char)
    if rows is None:
        raise KeyError(f"字型中沒有 {char!r}")
    return np.array([[cell == "1" for cell in row] for row in rows], dtype=bool)


def render_mask(text: str, gap: int = 1, padding: int = 2) -> np.ndarray:
    """把字串排成一條遮罩。

    ``gap`` 是字元間距；設成 0 就能製造「字元黏在一起」的情況，
    用來驗證切割層的等分切割。
    """
    glyphs = [glyph_mask(char) for char in text]
    if not glyphs:
        return np.zeros((0, 0), dtype=bool)

    width = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1) + padding * 2
    height = GLYPH_HEIGHT + padding * 2
    canvas = np.zeros((height, width), dtype=bool)

    x = padding
    for glyph in glyphs:
        canvas[padding : padding + glyph.shape[0], x : x + glyph.shape[1]] = glyph
        x += glyph.shape[1] + gap
    return canvas


def render_bgra(
    text: str,
    gap: int = 1,
    padding: int = 2,
    scale: int = 1,
    ink: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (24, 20, 16),
    noise: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """渲染成 BGRA 影像，用來測試包含二值化在內的完整管線。

    ``noise`` 是要撒進背景的單像素雜點數量 —— 真實畫面一定有雜訊，
    切割層必須能把這種雜點濾掉。
    """
    mask = render_mask(text, gap=gap, padding=padding)
    if scale > 1:
        mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)

    height, width = mask.shape
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., :3] = np.array(background, dtype=np.uint8)
    image[mask, :3] = np.array(ink, dtype=np.uint8)
    image[..., 3] = 255

    if noise > 0:
        rng = np.random.default_rng(seed)
        for _ in range(noise):
            y = int(rng.integers(0, height))
            x = int(rng.integers(0, width))
            image[y, x, :3] = np.array(ink, dtype=np.uint8)
    return image


def build_template_set():
    """用內建字型組出一份 TemplateSet，模擬校準完成後的狀態。"""
    from .vision.templates import TemplateSet

    templates = TemplateSet()
    for char in GLYPHS:
        templates.add(char, glyph_mask(char))
    templates.meta = {"source": "builtin-testfont"}
    return templates
