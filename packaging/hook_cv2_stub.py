"""執行期掛鉤：放一個 OpenCV 的替身進去。

``windows_capture``（WGC 擷取後端）在模組最上面就 ``import cv2``，但 cv2 只被它的
``save_as_image()`` 用到 —— 把畫面存成圖檔。這個專案從頭到尾只拿 ``frame_buffer``
的原始像素，一次都沒呼叫過那個方法。

為了一個用不到的功能把整包 OpenCV 塞進執行檔，會從 27MB 變成 65MB。所以這裡在
``windows_capture`` 被匯入之前先放一個替身，讓那行 import 過得去。真的有人呼叫到
需要 OpenCV 的功能時會收到一句講清楚的錯誤，而不是莫名其妙的 AttributeError。

**為什麼不能乾脆不裝 WGC**：WGC 是「遊戲全螢幕或被其他視窗蓋住時還能擷取」的唯一
辦法。少了它會退回螢幕 BitBlt，那條路會把蓋在上面的視窗一起拍進去。
"""

from __future__ import annotations

import sys
import types

_MESSAGE = (
    "這份打包沒有包含 OpenCV。windows_capture 只有「把畫面存成圖檔」的功能需要它，"
    "而這個程式不使用該功能 —— 會走到這裡代表有東西改了，請回報。"
)


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(_MESSAGE)


if "cv2" not in sys.modules:
    stub = types.ModuleType("cv2")
    stub.__doc__ = _MESSAGE
    stub.imwrite = _unavailable
    stub.imread = _unavailable
    sys.modules["cv2"] = stub
