"""把 ``assets/Logo.jpeg`` 做成 Windows 的 ``assets/icon.ico``。

為什麼要一支腳本而不是手動轉一次：圖示要同時出現在 16px 的工作列和 256px 的
檔案總管，縮圖品質差很多 —— 一般工具的縮放會把小尺寸弄糊。這裡用**面積平均**
縮圖（每個輸出像素取對應輸入區域的平均，含邊界的分數權重），小尺寸才看得清楚。
Logo 之後換了也只要重跑一次。

解碼 JPEG 借用 Windows 內建的 System.Drawing（透過 PowerShell）—— 這是建置期
腳本，不是執行期程式碼，所以不會因此多一個執行期相依套件。

用法::

    python scripts/make_icon.py
    python scripts/make_icon.py --preview    # 只輸出預覽，不寫 .ico
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapleexp.vision import pngio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SOURCE = ASSETS / "Logo.jpeg"

# 圖示尺寸。Windows 會從這些裡面挑最接近的；少了 16/32 的話工作列會拿大圖硬縮。
SIZES = (16, 24, 32, 48, 64, 128, 256)

# 圓角半徑佔邊長的比例。量自 Logo 本身（581px 寬、圓角約 83px）。
CORNER_RADIUS = 83 / 581

# 小尺寸改用「只有楓葉」的裁切。完整 Logo 在 48px 以下會糊成一團（「EXP TOOL」
# 那行字在那個尺寸本來就不可能讀），只留最有辨識度的楓葉反而清楚得多 ——
# 同一個圖示檔在不同尺寸放不同構圖是常見做法，Windows 會自己挑。
EMBLEM_MAX_SIZE = 48

# 楓葉在原圖裡的位置（量出來的：楓葉 x 178-396、y 159-379，文字從 y=411 開始）。
# 下緣刻意停在 400，免得把「MS」的字頭掃進來。
EMBLEM_BOX = (138, 156, 262)   # (上, 左, 邊長)
EMBLEM_CORNER_RADIUS = 0.14

_DECODE = r"""
Add-Type -AssemblyName System.Drawing
$bmp = [System.Drawing.Bitmap]::FromFile('{src}')
$rect = New-Object System.Drawing.Rectangle 0,0,$bmp.Width,$bmp.Height
$data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                      [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$bytes = New-Object byte[] ($data.Stride * $bmp.Height)
[System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
$bmp.UnlockBits($data)
[System.IO.File]::WriteAllBytes('{raw}', $bytes)
"$($bmp.Width),$($bmp.Height),$($data.Stride)" | Out-File -Encoding ascii '{meta}'
$bmp.Dispose()
"""


def decode(path: Path) -> np.ndarray:
    """解出 (h, w, 4) 的 RGBA。"""
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "image.raw"
        meta = Path(folder) / "image.txt"
        script = _DECODE.format(src=path, raw=raw, meta=meta)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
        )
        if not raw.exists():
            raise RuntimeError(
                f"解碼失敗：{result.stderr.strip() or result.stdout.strip()}"
            )
        width, height, stride = (int(v) for v in meta.read_text().strip().split(","))
        bgra = np.fromfile(raw, dtype=np.uint8).reshape(height, stride // 4, 4)[:, :width]
    return np.ascontiguousarray(bgra[..., [2, 1, 0, 3]])


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """補成正方形。

    Logo 的卡片是滿版的，而原圖上下各少了幾個像素；直接拉伸會讓圓角變形，所以
    用四角的底色補邊而不是縮放。
    """
    height, width = image.shape[:2]
    if height == width:
        return image
    side = max(height, width)
    out = np.zeros((side, side, 4), dtype=np.uint8)
    out[:] = image[0, 0]
    top = (side - height) // 2
    left = (side - width) // 2
    out[top : top + height, left : left + width] = image
    return out


def round_corners(image: np.ndarray, radius_ratio: float = CORNER_RADIUS) -> np.ndarray:
    """把圓角外面挖成透明。

    Logo 本身就是一張圓角卡片，但四角那一小塊是不透明的淺灰。留著的話，深色的
    工作列上會看到四個白色小角。
    """
    side = image.shape[0]
    radius = radius_ratio * side
    axis = np.arange(side) + 0.5
    # 每個座標離「圓角圓心所在那條線」多遠；只有兩軸都為正才落在角落區域
    offset = np.maximum(radius - axis, axis - (side - radius))
    dy = offset[:, None]
    dx = offset[None, :]
    corner = (dx > 0) & (dy > 0)
    distance = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
    # 1px 的過渡帶做反鋸齒，邊緣才不會有階梯
    alpha = np.where(corner, np.clip(radius + 0.5 - distance, 0.0, 1.0), 1.0)

    out = image.astype(np.float64)
    out[..., 3] *= alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def area_resize(image: np.ndarray, size: int) -> np.ndarray:
    """面積平均縮圖。

    每個輸出像素取它對應的輸入矩形的平均（邊界給分數權重），用積分影像一次算完。
    比雙線性取樣好：縮很多倍時，雙線性只看 4 個鄰居，中間的細節會整個跳過。
    """
    source = image.astype(np.float64)
    # 先乘上 alpha 再平均，透明區域的顏色才不會把邊緣染暗
    source[..., :3] *= source[..., 3:4] / 255.0

    height, width = source.shape[:2]
    integral = np.zeros((height + 1, width + 1, 4), dtype=np.float64)
    integral[1:, 1:] = source.cumsum(axis=0).cumsum(axis=1)

    ys = np.arange(size + 1) * (height / size)
    xs = np.arange(size + 1) * (width / size)

    low_y = np.floor(ys).astype(int)
    high_y = np.minimum(low_y + 1, height)
    frac_y = (ys - low_y)[:, None, None]
    rows = integral[low_y] * (1 - frac_y) + integral[high_y] * frac_y

    low_x = np.floor(xs).astype(int)
    high_x = np.minimum(low_x + 1, width)
    frac_x = (xs - low_x)[None, :, None]
    grid = rows[:, low_x] * (1 - frac_x) + rows[:, high_x] * frac_x

    total = grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1]
    out = total / ((height / size) * (width / size))

    alpha = out[..., 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., :3] = np.where(alpha > 0.5, out[..., :3] / (alpha / 255.0), 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def encode_ico(images: list[np.ndarray]) -> bytes:
    """把幾張 RGBA 包成 .ico。每一張都用 PNG 壓縮（Vista 以後都支援）。"""
    blobs = [pngio.encode_png(image) for image in images]
    header = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries = []
    for image, blob in zip(images, blobs):
        size = image.shape[0]
        entries.append(
            struct.pack(
                "<BBBBHHII",
                0 if size >= 256 else size,   # 256 在 .ico 裡要寫成 0
                0 if size >= 256 else size,
                0, 0, 1, 32, len(blob), offset,
            )
        )
        offset += len(blob)
    return header + b"".join(entries) + b"".join(blobs)


def checkerboard(shape: tuple[int, int]) -> np.ndarray:
    """預覽用的棋盤格底，這樣看得出哪裡是透明的。"""
    board = (
        np.arange(shape[0])[:, None] // 8 + np.arange(shape[1])[None, :] // 8
    ) % 2
    out = np.zeros((*shape, 4), dtype=np.uint8)
    out[..., :3] = np.where(board[..., None], 215, 245)
    out[..., 3] = 255
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="只輸出預覽 PNG")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(ASSETS))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"找不到 {source}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    square = pad_to_square(decode(source))
    original = round_corners(square)
    top, left, side = EMBLEM_BOX
    emblem = round_corners(
        np.ascontiguousarray(square[top : top + side, left : left + side]),
        EMBLEM_CORNER_RADIUS,
    )
    print(f"來源 {source.name}：{original.shape[1]}x{original.shape[0]}"
          f"（{EMBLEM_MAX_SIZE}px 以下改用楓葉裁切）")

    images = [
        area_resize(emblem if size <= EMBLEM_MAX_SIZE else original, size)
        for size in SIZES
    ]
    for size, image in zip(SIZES, images):
        (out / f"icon-{size}.png").write_bytes(pngio.encode_png(image))

    height = max(SIZES)
    strip = checkerboard((height, sum(SIZES) + 8 * len(SIZES)))
    x = 0
    for size, image in zip(SIZES, images):
        top = (height - size) // 2
        under = strip[top : top + size, x : x + size].astype(np.float64)
        over = image.astype(np.float64)
        weight = over[..., 3:4] / 255.0
        under[..., :3] = over[..., :3] * weight + under[..., :3] * (1 - weight)
        strip[top : top + size, x : x + size] = under.astype(np.uint8)
        x += size + 8
    (out / "icon-preview.png").write_bytes(pngio.encode_png(strip))

    if not args.preview:
        (out / "icon.ico").write_bytes(encode_ico(images))
        print(f"寫入 {out / 'icon.ico'}（{len(SIZES)} 種尺寸）")
    print(f"預覽：{out / 'icon-preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
