"""Win32 子模組。

匯入這個 package 就會立刻宣告 per-monitor DPI 感知。

這一步必須發生在**任何**視窗幾何查詢之前：在縮放 125% 的螢幕上，
還沒宣告感知時 GetClientRect 回報的是邏輯像素（1536x864），
擷取時拿到的卻是實體像素（1920x1080）—— 同一個行程裡兩套座標混用，
遮蔽判定與 ROI 換算都會錯。
"""

from .api import enable_dpi_awareness as _enable_dpi_awareness

DPI_MODE = _enable_dpi_awareness()
