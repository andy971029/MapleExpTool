"""設定檔與路徑管理。

所有校準結果（ROI、字元模板、門檻值）都存在 %LOCALAPPDATA%/MapleExpTracker 下，
不寫進專案目錄，方便重裝程式而不用重新校準。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# 路徑
# --------------------------------------------------------------------------- #

APP_NAME = "MapleExpTracker"


def data_dir() -> Path:
    """使用者資料目錄（可用 MAPLEEXP_HOME 覆寫，測試時很方便）。"""
    override = os.environ.get("MAPLEEXP_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / APP_NAME


def resource_path(relative: str) -> Path:
    """跟程式一起散佈的唯讀檔案（圖示等）的位置。

    打包成單一執行檔之後，這些資源會被解壓到一個暫存目錄（``sys._MEIPASS``），
    而不是跟 .exe 放在一起；用 ``__file__`` 往上推會指到不存在的路徑。
    """
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) if base else Path(__file__).resolve().parents[2]
    return root / relative


def icon_path() -> Path:
    return resource_path("assets/icon.ico")


def config_path() -> Path:
    return data_dir() / "config.json"


def templates_dir() -> Path:
    return data_dir() / "templates"


def db_path() -> Path:
    return data_dir() / "tracker.db"


def exp_table_path() -> Path:
    return data_dir() / "exp_table.json"


def debug_dir() -> Path:
    return data_dir() / "debug"


# --------------------------------------------------------------------------- #
# 資料結構
# --------------------------------------------------------------------------- #


# 錨點名稱 -> (水平比例, 垂直比例)
ANCHORS: dict[str, tuple[float, float]] = {
    "top-left": (0.0, 0.0),
    "top-center": (0.5, 0.0),
    "top-right": (1.0, 0.0),
    "center": (0.5, 0.5),
    "bottom-left": (0.0, 1.0),
    "bottom-center": (0.5, 1.0),
    "bottom-right": (1.0, 1.0),
}

DEFAULT_ANCHOR = "bottom-center"


@dataclass
class Roi:
    """畫面上的一塊區域，用「錨點 + 像素位移」表示。

    為什麼不用相對比例（0..1）：楓之谷的狀態列是**固定像素大小**、錨定在
    畫面底部中央的 UI，不會隨視窗變大而等比放大。用比例存，視窗一改大小
    ROI 就整個跑掉；用「距離底部中央幾個像素」存，縮放視窗後依然對準。

    ``dx``/``dy`` 是錨點到 ROI 左上角的位移（可為負），``w``/``h`` 是像素大小。
    """

    anchor: str = DEFAULT_ANCHOR
    dx: int = 0
    dy: int = 0
    w: int = 0
    h: int = 0

    def is_set(self) -> bool:
        return self.w > 0 and self.h > 0

    def anchor_point(self, width: int, height: int) -> tuple[int, int]:
        ratio_x, ratio_y = ANCHORS.get(self.anchor, ANCHORS[DEFAULT_ANCHOR])
        return int(round(ratio_x * width)), int(round(ratio_y * height))

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """換算成 (left, top, right, bottom)，並夾在畫面範圍內。"""
        origin_x, origin_y = self.anchor_point(width, height)
        left = origin_x + self.dx
        top = origin_y + self.dy
        right = left + self.w
        bottom = top + self.h
        left = max(0, min(left, max(0, width - 1)))
        top = max(0, min(top, max(0, height - 1)))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return left, top, right, bottom

    @classmethod
    def from_pixels(
        cls,
        rect: tuple[int, int, int, int],
        width: int,
        height: int,
        anchor: str = DEFAULT_ANCHOR,
    ) -> "Roi":
        if width <= 0 or height <= 0:
            raise ValueError("width/height 必須為正數")
        left, top, right, bottom = rect
        ratio_x, ratio_y = ANCHORS.get(anchor, ANCHORS[DEFAULT_ANCHOR])
        origin_x = int(round(ratio_x * width))
        origin_y = int(round(ratio_y * height))
        return cls(
            anchor=anchor,
            dx=left - origin_x,
            dy=top - origin_y,
            w=max(1, right - left),
            h=max(1, bottom - top),
        )


# 在 1920x1080 的 新楓之谷：經典版 狀態列實測到的位置，給校準介面當起始建議。
# 因為是「距離底部中央的像素位移」，視窗改大小後仍然對得上
# （狀態列是固定像素大小、錨定在底部中央的 UI）。
# dy/h 刻意避開狀態列面板上緣那條亮線：把它框進來會逼得二值化門檻必須拉很高，
# 結果連綠色的方括號與小數點都被濾掉，百分比就解析不出來了。
SUGGESTED_EXP_ROI = Roi(anchor="bottom-center", dx=153, dy=-38, w=81, h=15)


@dataclass
class CaptureConfig:
    """視窗擷取設定。"""

    # 視窗標題的比對字串（子字串比對，不分大小寫）。
    window_title_contains: str = "新楓之谷"
    # 若標題比對不到，可直接指定 exe 名稱。
    process_name: str = "Maplestory_Classic.exe"
    # "auto" | "printwindow" | "screen"
    backend: str = "auto"


@dataclass
class ReaderConfig:
    """字元辨識設定。"""

    # 只有經驗值需要 ROI。等級是從狀態列的橘色方塊自動定位的（vision/identity.py），
    # 不需要、也沒有對應的設定。
    exp_roi: Roi = field(default_factory=Roi)
    # 亮度二值化門檻（0-255）。校準時用滑桿調。
    threshold: int = 150
    # True = 取暗色文字（亮底），False = 取亮色文字（暗底）。
    invert: bool = False
    # 指定文字顏色（BGR）時改用色距判斷，而不是單純的亮度門檻。
    # 狀態列背景有紋理或漸層時，色鍵通常乾淨得多。
    ink_color: list[int] | None = None
    color_tolerance: int = 90
    # 寬度明顯超過中位數的連通區塊是否要等分切開（字元黏在一起時需要）。
    split_wide_glyphs: bool = True
    # 字元切割參數
    min_glyph_width: int = 2
    min_glyph_height: int = 3
    # 模板比對相似度下限（0..1）；低於此值視為辨識失敗。
    match_min_score: float = 0.80
    # 兩個候選字元分數太接近時視為不可靠（避免 8/9 互換）。
    match_min_margin: float = 0.02


@dataclass
class TrackerConfig:
    """經驗值追蹤邏輯設定。"""

    # 取樣間隔（秒）。EXP 條是 UI，1 秒已經很夠。
    sample_interval: float = 1.0
    # 連續辨識失敗幾次才判定為「中斷」。
    miss_to_pause: int = 3
    # 經驗值停滯超過這麼多秒就停止累計活躍時間（走路、聊天、掛機都算）。
    # 設 0 可關閉。注意：等級很高、單隻怪間隔超過這個秒數時會把速率算得偏高，
    # 這種情況要把它調大。
    idle_pause_sec: float = 10.0
    # 中斷時間 <= 此值（秒）視為短暫遮蔽，經驗差值與時間都照算；
    # 超過則兩者都捨棄（無法判斷是掛機還是只是被遮住）。
    max_bridge_sec: float = 30.0
    # 單次取樣的經驗增量若超過「本等級所需經驗 × 此比例」，判定為辨識錯誤。
    outlier_level_frac: float = 0.25
    # 異常轉換（下降／升級）是否需要下一筆取樣確認後才採信。
    # 開啟可過濾掉單格辨識雜訊，代價是延遲一次取樣。
    confirm_surprises: bool = True
    # 統計視窗（秒）。
    rate_windows: list[int] = field(default_factory=lambda: [300, 900, 3600])
    # 用哪個視窗估算升級剩餘時間。
    eta_window: int = 900


# 0.1 版的浮窗透明度。太透了 —— 遊戲場景亮的時候字會跟背景混在一起看不清楚。
# 使用者設定檔裡若還是這個值就視為「沒調過」，換成新的預設。
LEGACY_OVERLAY_ALPHA = 0.88


@dataclass
class UiConfig:
    overlay_alpha: float = 0.96
    overlay_x: int = 40
    overlay_y: int = 40
    overlay_scale: float = 1.0
    always_on_top: bool = True
    # 縮小模式：只留一排（狀態燈、EXP/h、升級剩餘、按鈕）。
    overlay_compact: bool = False


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    # 是否把每一格辨識失敗的 ROI 影像存到 debug 目錄。
    save_debug_frames: bool = False
    # 啟動時到 GitHub 查有沒有新版（只有打包成執行檔的版本會做）。
    check_updates: bool = True
    # 按過「略過這版」的版本號；同一版不再提示，更新的版本照提。
    skipped_update: str = ""

    # ---------------- 存取 ----------------

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        target = path or config_path()
        if not target.exists():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        cfg = _from_dict(cls, raw)
        if cfg.ui.overlay_alpha == LEGACY_OVERLAY_ALPHA:
            cfg.ui.overlay_alpha = UiConfig().overlay_alpha
        return cfg

    def is_calibrated(self) -> bool:
        return self.reader.exp_roi.is_set()


# --------------------------------------------------------------------------- #
# dataclass 反序列化（容忍舊版欄位缺失／多餘）
# --------------------------------------------------------------------------- #


def _from_dict(cls: type, raw: Any) -> Any:
    if not isinstance(raw, dict):
        return cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        kwargs[f.name] = raw[f.name]
    # `from __future__ import annotations` 讓 f.type 變成字串，沒辦法用 is_dataclass
    # 判斷哪些欄位是巢狀結構，所以用名稱對照表。
    nested = {
        "capture": CaptureConfig,
        "reader": ReaderConfig,
        "tracker": TrackerConfig,
        "ui": UiConfig,
        "exp_roi": Roi,
    }
    for name, sub_cls in nested.items():
        if name in kwargs and isinstance(kwargs[name], dict):
            raw_sub = kwargs[name]
            if sub_cls is Roi and "anchor" not in raw_sub:
                # 0.1 之前的相對比例格式；沿用會得到看似合法但完全錯誤的 ROI。
                kwargs[name] = Roi()
                continue
            kwargs[name] = _from_dict(sub_cls, raw_sub)
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
