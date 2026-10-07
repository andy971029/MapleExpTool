"""最小的 PNG 編碼器（只用標準庫）。

有兩個用途，所以值得自己寫而不是拉一個影像套件進來：

1. 把辨識失敗的 ROI 存成圖檔，方便事後用眼睛看是哪裡出錯。
2. 餵給 Tk 的 ``PhotoImage(data=...)`` 當校準介面的預覽圖 —— Tk 9 原生支援 PNG，
   所以整個 GUI 也不需要 Pillow。
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


# PNG 的顏色型態：2 = 真彩色，6 = 真彩色加 alpha。
_COLOUR_TYPE = {3: 2, 4: 6}


def encode_png(image: np.ndarray) -> bytes:
    """把 (h, w, 3) 的 RGB 或 (h, w, 4) 的 RGBA uint8 陣列編成 PNG。

    支援 RGBA 是為了畫程式圖示 —— 圖示一定要有透明背景。
    """
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in _COLOUR_TYPE:
        raise ValueError("預期為 (h, w, 3) 的 RGB 或 (h, w, 4) 的 RGBA 陣列")

    height, width, channels = array.shape
    # 每一列前面要加一個 filter byte（0 = None）。
    rows = np.hstack(
        [np.zeros((height, 1), dtype=np.uint8), array.reshape(height, width * channels)]
    )
    header = struct.pack(
        ">IIBBBBB", width, height, 8, _COLOUR_TYPE[channels], 0, 0, 0
    )
    return (
        _SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    """擷取層給的是 BGRA；PNG 要 RGB。"""
    return np.ascontiguousarray(bgra[..., 2::-1])


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """二值遮罩轉成黑白圖（ink 為白）。"""
    value = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    return np.repeat(value[:, :, None], 3, axis=2)


def scale_nearest(rgb: np.ndarray, factor: int) -> np.ndarray:
    """最近鄰放大。放大二值遮罩看字形時必備（一個像素太小看不出來）。"""
    if factor <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def to_tk_data(rgb: np.ndarray) -> str:
    """編成 Tk ``PhotoImage(data=...)`` 可以吃的 base64 字串。"""
    return base64.b64encode(encode_png(rgb)).decode("ascii")
