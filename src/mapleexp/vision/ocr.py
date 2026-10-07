"""中文辨識，用 Windows 內建的 OCR 引擎。

為什麼是這個而不是 Tesseract／PaddleOCR：Windows 10/11 本身就帶 OCR 引擎，
這台機器上 ``繁體中文 (台灣)`` 現成可用 —— **不用下載任何模型、不用裝外部執行檔**，
完全符合「開箱即用」。只需要 ``winrt-Windows.Media.Ocr`` 這個純綁定套件。

前處理很關鍵，實測差很多，而且關鍵不在放大倍率，在**怎麼把字從底色裡挑出來**。

為什麼用「解混合」而不是亮度門檻
--------------------------------

這個遊戲的 UI 是 2D 點陣圖，小地圖名稱區只用三種顏色：

====================  ========================  ==========
角色                  實測顏色（BGR）           佔比
====================  ========================  ==========
底色（淺鋼藍面板）    (204, 187, 153)           36.9%
文字                  (255, 255, 255)           2.6%
陰影                  (0, 0, 0)                 0.4%
====================  ========================  ==========

但 1421 個像素裡有 **634 種顏色** —— 一半以上是這三色的混合（抗鋸齒）。這就是
亮度門檻會失敗的原因：白字和黑陰影各半混出來的灰，亮度**剛好跟底色差不多**，
門檻法會在筆畫中間挖出洞來。字越小、筆畫越密，被挖掉的越多。

既然調色盤固定，每個像素就是三色的線性混合，可以逐像素把比例解回來：

    P = a*文字 + b*陰影 + c*底色,  a + b + c = 1

三條方程式（B/G/R）兩個未知數，最小平方解出來的 ``a`` 就是**文字的覆蓋率**，
等於把抗鋸齒的次像素資訊原封不動還原。底色不寫死，從畫面上面積最大的顏色量出來，
所以同一套程式對暗底（狀態列）、亮底（小地圖）、純白底（面板標題）都適用。
（三個顏色剛好共線時 —— 純灰階的 UI —— 拆分在數學上不唯一，那時退回單純的
方向投影，也就是舊的亮度做法，因為那個情況下顏色本來就沒有多餘的資訊。）

效果（實機 18 幀，地圖「迷霧森林／奇幻村」）：

====================  ==========================
做法                  結果
====================  ==========================
亮度門檻              ``'幻村'``（18/18 幀都只有這樣）
解混合                ``'迷霧森林
奇幻村'``（18/18 幀全對）
====================  ==========================

連 19 劃的「霧」都讀得出來。在這之前我已經量到「地區那行 90 次嘗試 0 次讀到」，
一度以為是字級的硬限制 —— 不是，是前處理把資訊丟掉了。

其餘兩個仍然必要的步驟：**雙線性放大**（最近鄰會把筆畫變成鋸齒方塊）、**四周留白**
（沒有邊距時引擎常常整張放棄）。倍率 3/4/6 各跑一次取最長，因為小字對倍率很敏感，
沒有哪個倍率永遠最好。
"""

from __future__ import annotations

import asyncio
import re

import numpy as np


try:  # pragma: no cover - 取決於環境
    import winrt.windows.media.ocr as _ocr
    import winrt.windows.storage.streams as _streams
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap

    OCR_AVAILABLE = True
    OCR_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _ocr = None
    OCR_AVAILABLE = False
    OCR_IMPORT_ERROR = str(exc)

_engine = None
_engine_tried = False

# 中文辨識會在每個字之間插一個空白，那不是詞的分界，直接拿掉。
_SPACES = re.compile(r"\s+")


def engine():
    """取得（並快取）OCR 引擎。沒有可用語言時回傳 None。"""
    global _engine, _engine_tried
    if _engine_tried:
        return _engine
    _engine_tried = True
    if not OCR_AVAILABLE:
        return None
    try:
        _engine = _ocr.OcrEngine.try_create_from_user_profile_languages()
        if _engine is None:
            # 使用者設定的語言沒有 OCR 支援時，挑第一個可用的。
            languages = list(_ocr.OcrEngine.available_recognizer_languages)
            if languages:
                _engine = _ocr.OcrEngine.try_create_from_language(languages[0])
    except Exception:
        _engine = None
    return _engine


def language_tag() -> str:
    instance = engine()
    try:
        return instance.recognizer_language.language_tag if instance else ""
    except Exception:
        return ""


def bilinear(src: np.ndarray, scale: int) -> np.ndarray:
    """雙線性放大。

    最近鄰放大會把筆畫變成鋸齒方塊，OCR 的辨識率掉得非常明顯；
    雙線性把筆畫的連續性還原回來，是這裡成敗的關鍵。
    """
    height, width = src.shape
    ys = (np.arange(height * scale) + 0.5) / scale - 0.5
    xs = (np.arange(width * scale) + 0.5) / scale - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, width - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    return (
        src[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
        + src[np.ix_(y0, x1)] * (1 - wy) * wx
        + src[np.ix_(y1, x0)] * wy * (1 - wx)
        + src[np.ix_(y1, x1)] * wy * wx
    )


# 底色往文字方向讓開這麼多，免得底色本身的雜訊被當成淡淡的墨水。
# 實測 0 跟 4 一樣好，8 開始掉字，所以取 4 留點餘裕。
# 這個遊戲 UI 的文字調色盤：純白的字配純黑的陰影（面板標題則相反）。
WHITE = (255.0, 255.0, 255.0)
BLACK = (0.0, 0.0, 0.0)

# 找底色時先把顏色量化成這麼粗的格子。底色偶爾有輕微漸層，直接找精確眾數會碎掉。
COLOUR_BIN = 8

# 文字跟底色至少要差這麼多才解得動；差太少時解出來的全是雜訊。
MIN_SEPARATION = 24.0


def _parallel(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-3) -> bool:
    """兩個顏色方向是不是共線（夾角接近 0 或 180 度）。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return True
    return abs(abs(float(a @ b)) / (na * nb) - 1.0) < tolerance


def dominant_colour(bgra: np.ndarray, bins: int = COLOUR_BIN) -> np.ndarray:
    """這塊畫面裡面積最大的顏色，也就是底色。

    先量化成粗格子找眾數，再回傳那一格裡所有像素的平均：直接找精確眾數的話，
    只要底色有一點漸層就會被打散成好幾個票數相近的顏色。
    """
    flat = bgra[..., :3].reshape(-1, 3).astype(np.int32)
    quantised = flat // max(1, bins)
    code = (quantised[:, 0] << 16) | (quantised[:, 1] << 8) | quantised[:, 2]
    values, counts = np.unique(code, return_counts=True)
    chosen = values[int(np.argmax(counts))]
    return flat[code == chosen].mean(axis=0).astype(np.float32)


def coverage(
    bgra: np.ndarray,
    text=WHITE,
    shadow=BLACK,
    background=None,
) -> np.ndarray:
    """逐像素解出**文字的覆蓋率**（0~1）。

    把每個像素當成 文字／陰影／底色 三色的線性混合，解出文字那一項的比例。
    詳細理由見模組說明 —— 簡單說，這樣才拿得回抗鋸齒裡的次像素資訊，而亮度
    門檻會把「白字和黑陰影混出來的灰」誤判成底色。
    """
    base = (
        dominant_colour(bgra) if background is None
        else np.asarray(background, dtype=np.float32)
    )
    to_text = np.asarray(text, np.float32) - base
    to_shadow = np.asarray(shadow, np.float32) - base
    if np.linalg.norm(to_text) < MIN_SEPARATION:
        # 文字跟底色根本同色，沒有東西好解。
        return np.zeros(bgra.shape[:2], dtype=np.float32)

    centred = (bgra[..., :3].astype(np.float32) - base).reshape(-1, 3)
    if _parallel(to_text, to_shadow):
        # 三個顏色在 RGB 空間裡共線（純灰階的 UI 就是這樣）。這時候「這個像素有
        # 幾成是字、幾成是陰影」在數學上就是不唯一的 —— 任何解法都只是在猜。
        # 退回最直觀的約定：比底色偏文字方向的算字，偏陰影方向的算 0。
        alpha = centred @ to_text / float(to_text @ to_text)
    else:
        try:
            inverse = np.linalg.pinv(np.stack([to_text, to_shadow], axis=1))
        except np.linalg.LinAlgError:  # pragma: no cover - 理論上到不了
            return np.zeros(bgra.shape[:2], dtype=np.float32)
        alpha = centred @ inverse[0]
    return np.clip(alpha.reshape(bgra.shape[:2]), 0.0, 1.0)


def prepare(bgra: np.ndarray, scale: int = 4, bright_text: bool = True,
            margin: int = 28) -> np.ndarray:
    """把遊戲畫面的一塊整理成 OCR 比較讀得動的樣子（黑字白底、放大、留白）。

    *bright_text* 說的是文字跟陰影哪個是亮的：狀態列與小地圖是白字黑陰影，
    面板標題是純白底的黑字，所以反過來。
    """
    text, shadow = (WHITE, BLACK) if bright_text else (BLACK, WHITE)
    alpha = coverage(bgra, text=text, shadow=shadow)
    # 先解混合再放大：解混合要用原始像素的顏色關係，插值過的顏色不在調色盤上。
    big = bilinear(alpha, max(1, scale))
    scaled = np.clip((1.0 - big) * 255.0, 0, 255).astype(np.uint8)
    padded = np.full(
        (scaled.shape[0] + margin * 2, scaled.shape[1] + margin * 2), 255, dtype=np.uint8
    )
    padded[margin : margin + scaled.shape[0], margin : margin + scaled.shape[1]] = scaled
    return padded


def recognize(bgra: np.ndarray, scale: int = 4, bright_text: bool = True) -> str:
    """辨識一塊畫面上的文字。讀不到就回空字串 —— 絕不猜。"""
    instance = engine()
    if instance is None or bgra is None or bgra.size == 0:
        return ""
    try:
        gray = prepare(bgra, scale=scale, bright_text=bright_text)
        image = np.zeros((*gray.shape, 4), dtype=np.uint8)
        for channel in range(3):
            image[..., channel] = gray
        image[..., 3] = 255
        image = np.ascontiguousarray(image)

        writer = _streams.DataWriter()
        writer.write_bytes(image.tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            writer.detach_buffer(),
            BitmapPixelFormat.BGRA8,
            image.shape[1],
            image.shape[0],
        )
        result = asyncio.run(instance.recognize_async(bitmap))
        return tidy(lines_of(result))
    except Exception:
        return ""


def recognize_best(bgra: np.ndarray, scales=(3, 4, 6), bright_text: bool = True) -> str:
    """幾個放大倍率都試，取字數最多的結果。

    小字的辨識率對倍率很敏感，而且沒有一個倍率永遠最好；試幾個再挑，
    比挑一個「通常還行」的值穩定得多。
    """
    best = ""
    for scale in scales:
        text = recognize(bgra, scale=scale, bright_text=bright_text)
        if len(text) > len(best):
            best = text
    return best


def lines_of(result) -> str:
    """從 OCR 結果取出**保留行結構**的文字。

    這裡踩過第二個坑：``OcrResult.text`` 把所有東西用空白串成一條，**裡面沒有
    換行**。而中文辨識本來就會在每個字之間插空白，所以把空白清掉之後，視覺上
    的兩行就黏成一條了 —— 小地圖的「地區 / 地圖名」因此分不開。

    引擎其實有給行結構（``result.lines``，連每個字的座標都有），只是 ``.text``
    這個便利屬性把它攤平了。所以改成自己走 ``lines`` 再用換行接起來。
    """
    try:
        lines = [line.text for line in result.lines]
    except Exception:
        lines = []
    if lines:
        return "\n".join(lines)
    try:
        return result.text or ""
    except Exception:
        return ""


def tidy(text: str) -> str:
    """拿掉中文 OCR 在每個字之間插的空白；換行保留。

    換行是 :func:`lines_of` 依照引擎給的行結構補上的，對小地圖面板很關鍵：
    第一行是地區、第二行是地圖名，分開才比對得準。
    """
    lines = [_SPACES.sub("", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)
