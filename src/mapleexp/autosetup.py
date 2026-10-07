"""開箱即用的自動設定。

目標：第一次執行什麼都不用做 —— 找到遊戲視窗、自己在畫面上認出經驗值欄位、
用內建模板開始算。校準精靈退居備案，只在自動設定失敗時才需要。

做得到的原因有兩個：
* 字元模板是從這個遊戲的狀態列實機採集的，內建在套件裡，不用使用者標。
* ROI 由 :mod:`mapleexp.vision.locate` 掃描畫面自己找，不用使用者框。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as config_module
from .config import Config
from .vision.locate import LocateResult, locate_exp_field
from .vision.templates import TemplateSet
from .win32.capture import CaptureError, WindowCapturer

REQUIRED_CHARS = set("0123456789")


def builtin_templates_path() -> Path:
    return Path(__file__).with_name("builtin_templates.json")


def load_templates(user_path: Path | None = None) -> tuple[TemplateSet, str]:
    """載入字元模板。回傳 (模板, 來源說明)。

    使用者自己校準過的優先；沒有或不完整就用內建的 —— 內建模板是從這個遊戲
    的狀態列實機採集的，多數人根本不需要自己標。
    """
    path = user_path or (config_module.templates_dir() / "digits.json")
    user = TemplateSet.load(path)
    if len(user) and not (REQUIRED_CHARS - set("".join(user.chars()))):
        return user, f"使用者校準（{path}）"

    builtin = TemplateSet.load(builtin_templates_path())
    if len(builtin):
        return builtin, "內建模板"
    return user, f"使用者校準（{path}）"


@dataclass
class AutoSetup:
    """自動設定的結果，給診斷訊息用。"""

    located: LocateResult | None
    template_source: str
    roi_source: str
    message: str

    @property
    def ok(self) -> bool:
        return self.roi_source != "失敗"


def auto_configure(
    cfg: Config,
    capturer: WindowCapturer,
    templates: TemplateSet,
    force: bool = False,
    template_source: str = "",
) -> AutoSetup:
    """必要時自動設定 ROI 與二值化參數。

    ``force=False`` 時，已經有 ROI 就直接沿用（使用者校準過的優先）。
    """
    if cfg.reader.exp_roi.is_set() and not force:
        return AutoSetup(
            located=None,
            template_source=template_source,
            roi_source="既有設定",
            message="沿用已儲存的 ROI",
        )

    try:
        frame = capturer.capture_client()
    except CaptureError as exc:
        return AutoSetup(None, template_source, "失敗", f"擷取畫面失敗：{exc}")

    found = locate_exp_field(frame.pixels, templates)
    if found is None:
        return AutoSetup(
            None,
            template_source,
            "失敗",
            "在畫面上找不到經驗值欄位。請確認狀態列沒有被遮住、"
            "或改用 `calibrate` 手動指定。",
        )

    cfg.reader.exp_roi = found.roi
    cfg.reader.threshold = found.threshold
    cfg.reader.invert = found.invert
    cfg.reader.ink_color = None

    detail = f"自動找到經驗值欄位 {found.text!r} @ {found.rect}"
    if not found.confident:
        detail += "（讀不到百分比，升級接續會不準，建議手動校準 ROI）"
    return AutoSetup(found, template_source, "自動偵測", detail)
