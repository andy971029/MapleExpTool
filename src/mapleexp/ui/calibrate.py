"""校準精靈（Tkinter）。

五個步驟：選視窗 -> 框 ROI -> 調二值化 -> 標字元 -> 驗證。

字元標記刻意做成「持續採集」而不是「一次湊齊 0-9」：
畫面上當下的經驗值只會包含其中幾個數字，要湊齊十個數字得等數字變化。
所以這一步會一直輪詢畫面，把沒見過的字形排進佇列讓人逐個標記，
邊打怪邊標，幾十秒就收滿了。
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from .. import config as config_module
from ..config import (
    ANCHORS,
    DEFAULT_ANCHOR,
    SUGGESTED_EXP_ROI,
    Config,
    Roi,
)
from ..core.reading import parse_exp_text
from ..vision import pngio
from ..vision.preprocess import binarize, suggest_binarization
from ..vision.segment import segment
from ..vision.templates import TemplateSet
from ..win32.capture import CaptureError, WindowCapturer
from ..win32.windows import WindowInfo, list_windows

DIGITS = "0123456789"
SYMBOLS = ".[]%,"
PREVIEW_MAX_WIDTH = 1100
PREVIEW_MAX_HEIGHT = 620
ROI_ZOOM = 4
GLYPH_ZOOM = 10
HARVEST_INTERVAL_MS = 700


class CalibrationApp(tk.Tk):
    def __init__(self, cfg: Config, templates_path: Path) -> None:
        super().__init__()
        self.title("楓之谷經驗計算器 — 校準")
        self.geometry("1180x820")
        self.minsize(900, 640)

        self.cfg = cfg
        self.templates_path = templates_path
        self.templates = TemplateSet.load(templates_path)

        self.window: WindowInfo | None = None
        self.capturer: WindowCapturer | None = None

        self._frame: np.ndarray | None = None       # 完整 client area
        self._frame_scale = 1.0
        self._photo: tk.PhotoImage | None = None
        self._drag_start: tuple[int, int] | None = None
        self._rect_id: int | None = None
        self._target_roi = "exp"                     # "exp" | "level"

        self._harvest_queue: list[np.ndarray] = []
        self._harvest_job: str | None = None
        self._verify_job: str | None = None

        self._build_layout()
        self._goto_window_step()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # 版面
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        self.header = ttk.Label(self, text="", font=("Segoe UI", 13, "bold"))
        self.header.pack(anchor="w", padx=14, pady=(12, 2))

        self.hint = ttk.Label(self, text="", wraplength=1120, justify="left")
        self.hint.pack(anchor="w", padx=14, pady=(0, 8))

        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, padx=14)

        self.controls = ttk.Frame(self)
        self.controls.pack(fill="x", padx=14, pady=10)

        self.status = ttk.Label(self, text="", foreground="#555")
        self.status.pack(anchor="w", padx=14, pady=(0, 10))

    def _reset_panels(self, header: str, hint: str) -> None:
        for widget in self.body.winfo_children():
            widget.destroy()
        for widget in self.controls.winfo_children():
            widget.destroy()
        self._cancel_jobs()
        self.header.config(text=header)
        self.hint.config(text=hint)

    def _cancel_jobs(self) -> None:
        for attr in ("_harvest_job", "_verify_job"):
            job = getattr(self, attr)
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.config(text=text, foreground="#b00020" if error else "#555")

    # ------------------------------------------------------------------ #
    # 步驟 1：選視窗
    # ------------------------------------------------------------------ #

    def _goto_window_step(self) -> None:
        self._reset_panels(
            "步驟 1 / 5 — 選擇遊戲視窗",
            "請先讓遊戲跑在視窗模式（不要獨占全螢幕），然後在下面挑出遊戲視窗。"
            "清單依 client area 大小排序，遊戲通常在最上面。",
        )

        self.window_list = tk.Listbox(self.body, height=16, font=("Consolas", 10))
        self.window_list.pack(fill="both", expand=True, pady=(4, 0))

        ttk.Button(self.controls, text="重新整理", command=self._refresh_windows).pack(
            side="left"
        )
        ttk.Button(self.controls, text="下一步", command=self._choose_window).pack(
            side="right"
        )
        self._refresh_windows()

    def _refresh_windows(self) -> None:
        self._windows = list_windows()
        self.window_list.delete(0, tk.END)
        preferred = -1
        needle = (self.cfg.capture.window_title_contains or "").casefold()
        for index, info in enumerate(self._windows):
            self.window_list.insert(tk.END, info.label)
            if preferred < 0 and needle and needle in info.title.casefold():
                preferred = index
            if preferred < 0 and info.process.casefold() == self.cfg.capture.process_name.casefold():
                preferred = index
        if preferred >= 0:
            self.window_list.selection_set(preferred)
            self.window_list.see(preferred)
        self._set_status(f"找到 {len(self._windows)} 個視窗")

    def _choose_window(self) -> None:
        selection = self.window_list.curselection()
        if not selection:
            self._set_status("請先選一個視窗", error=True)
            return
        info = self._windows[selection[0]]
        self.window = info
        if self.capturer is not None:
            self.capturer.close()
        self.capturer = WindowCapturer(info.hwnd, self.cfg.capture.backend)
        # 記住辨識特徵，之後追蹤時自動找同一個視窗。
        if info.title:
            self.cfg.capture.window_title_contains = info.title[:24]
        if info.process:
            self.cfg.capture.process_name = info.process
        self._goto_roi_step()

    # ------------------------------------------------------------------ #
    # 步驟 2：框 ROI
    # ------------------------------------------------------------------ #

    def _goto_roi_step(self) -> None:
        self._reset_panels(
            "步驟 2 / 5 — 框出經驗值區域",
            "在畫面上拖出一個矩形，框住狀態列的經驗值數字。"
            "框得「剛好包住數字」最理想 —— 盡量不要框進旁邊的中文字或圖示。\n"
            "如果狀態列同時顯示絕對值與百分比（例如 12345[30.25%]），"
            "請把兩者都框進去：有了百分比就能自動推算升級所需經驗，"
            "完全不需要內建等級經驗表。",
        )

        self.canvas = tk.Canvas(self.body, background="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        row1 = ttk.Frame(self.controls)
        row1.pack(fill="x")

        ttk.Label(row1, text="錨點").pack(side="left", padx=(0, 4))
        self._anchor = tk.StringVar(value=DEFAULT_ANCHOR)
        ttk.Combobox(
            row1,
            textvariable=self._anchor,
            values=list(ANCHORS),
            state="readonly",
            width=14,
        ).pack(side="left")
        ttk.Label(
            row1,
            text="（狀態列錨定在底部中央，用 bottom-center 可在縮放視窗後仍然對準）",
            foreground="#666",
        ).pack(side="left", padx=(8, 0))

        row2 = ttk.Frame(self.controls)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Button(row2, text="自動偵測", command=self._auto_locate).pack(side="left")
        ttk.Button(
            row2, text="套用建議位置", command=self._apply_suggested_roi
        ).pack(side="left", padx=(8, 0))
        ttk.Button(row2, text="重新擷取畫面", command=self._grab_frame).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(row2, text="上一步", command=self._goto_window_step).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(row2, text="下一步", command=self._finish_roi_step).pack(side="right")
        self._grab_frame()

    def _auto_locate(self) -> None:
        """讓程式自己在畫面上找經驗值欄位 —— 跟 `track` 第一次執行時做的事一樣。"""
        if self._frame is None:
            self._set_status("請先擷取畫面", error=True)
            return
        from ..vision.locate import locate_exp_field

        found = locate_exp_field(self._frame, self.templates)
        if found is None:
            self._set_status(
                "自動偵測失敗。請改用「套用建議位置」或自己拖曳框選。", error=True
            )
            return
        self.cfg.reader.exp_roi = found.roi
        self.cfg.reader.threshold = found.threshold
        self.cfg.reader.invert = found.invert
        self.cfg.reader.ink_color = None
        self._anchor.set(found.roi.anchor)
        self._set_status(f"自動偵測到 {found.text!r} @ {found.rect}")
        self._render_frame()

    def _apply_suggested_roi(self) -> None:
        """套用在 1920x1080 狀態列實測到的位置，當作微調的起點。"""
        self.cfg.reader.exp_roi = Roi(**vars(SUGGESTED_EXP_ROI))
        self._set_status(
            "已套用經驗值 ROI 建議位置（實測自 1920x1080 狀態列）。"
            "請確認框線剛好包住 112141[26.18%] 這串，必要時重新拖曳。"
        )
        self._anchor.set(DEFAULT_ANCHOR)
        self._render_frame()

    def _grab_frame(self) -> None:
        """擷取整個 client area。擷取期間先把校準視窗藏起來，避免遮住遊戲。"""
        if self.capturer is None:
            return
        self.withdraw()
        self.update()
        time.sleep(0.18)
        try:
            frame = self.capturer.capture_client()
        except CaptureError as exc:
            self.deiconify()
            self._set_status(f"擷取失敗：{exc}", error=True)
            return
        finally:
            self.deiconify()
            self.lift()

        self._frame = frame.pixels
        self._render_frame()
        self._set_status(
            f"擷取成功 {frame.width}x{frame.height}（後端：{frame.backend}）。"
            "請拖曳框選經驗值區域。"
        )

    def _render_frame(self) -> None:
        if self._frame is None:
            return
        rgb = pngio.bgra_to_rgb(self._frame)
        height, width = rgb.shape[:2]
        step = 1
        while (width // step) > PREVIEW_MAX_WIDTH or (height // step) > PREVIEW_MAX_HEIGHT:
            step += 1
        preview = rgb[::step, ::step]
        self._frame_scale = 1.0 / step
        self._photo = tk.PhotoImage(data=pngio.to_tk_data(preview))
        self.canvas.delete("all")
        self.canvas.config(width=preview.shape[1], height=preview.shape[0])
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._rect_id = None
        self._draw_existing_rois()

    def _draw_existing_rois(self) -> None:
        if self._frame is None:
            return
        height, width = self._frame.shape[:2]
        roi = self.cfg.reader.exp_roi
        if not roi.is_set():
            return
        left, top, right, bottom = roi.to_pixels(width, height)
        self.canvas.create_rectangle(
            left * self._frame_scale,
            top * self._frame_scale,
            right * self._frame_scale,
            bottom * self._frame_scale,
            outline="#00e5ff",
            width=2,
            tags="exp",
        )

    def _on_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ff5252", width=2
        )

    def _on_drag_move(self, event) -> None:
        if self._drag_start is None or self._rect_id is None:
            return
        self.canvas.coords(
            self._rect_id, self._drag_start[0], self._drag_start[1], event.x, event.y
        )

    def _on_drag_end(self, event) -> None:
        if self._drag_start is None or self._frame is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        left, right = sorted((x0, event.x))
        top, bottom = sorted((y0, event.y))
        if right - left < 4 or bottom - top < 4:
            self._set_status("框選範圍太小，請重新拖曳", error=True)
            return

        height, width = self._frame.shape[:2]
        scale = 1.0 / self._frame_scale
        rect = (
            max(0, int(left * scale)),
            max(0, int(top * scale)),
            min(width, int(right * scale)),
            min(height, int(bottom * scale)),
        )
        self.cfg.reader.exp_roi = Roi.from_pixels(
            rect, width, height, anchor=self._anchor.get()
        )
        self._set_status(f"經驗值 ROI 已設定：{rect}")
        self._render_frame()

    def _finish_roi_step(self) -> None:
        if not self.cfg.reader.exp_roi.is_set():
            self._set_status("請先框出經驗值 ROI", error=True)
            return
        self._goto_threshold_step()

    # ------------------------------------------------------------------ #
    # 步驟 3：二值化
    # ------------------------------------------------------------------ #

    def _goto_threshold_step(self) -> None:
        self._reset_panels(
            "步驟 3 / 5 — 調整二值化",
            "目標：右邊的黑白圖裡，數字要完整、清楚、彼此分開，背景要全黑。\n"
            "如果背景有紋理或漸層，單純調亮度門檻可能永遠乾淨不了 —— "
            "這時改用「色鍵」：在左邊原圖上點一下數字的顏色，程式會改用色距判斷。",
        )

        top = ttk.Frame(self.body)
        top.pack(fill="both", expand=True)

        left_panel = ttk.LabelFrame(top, text="原圖（點一下數字可取色）")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.roi_canvas = tk.Canvas(left_panel, background="#111", height=150)
        self.roi_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.roi_canvas.bind("<Button-1>", self._pick_ink_colour)

        right_panel = ttk.LabelFrame(top, text="二值化結果（白 = 文字）")
        right_panel.pack(side="left", fill="both", expand=True)
        self.mask_canvas = tk.Canvas(right_panel, background="#111", height=150)
        self.mask_canvas.pack(fill="both", expand=True, padx=6, pady=6)

        self.segment_label = ttk.Label(self.body, text="", font=("Consolas", 10))
        self.segment_label.pack(anchor="w", pady=(8, 0))

        row = ttk.Frame(self.controls)
        row.pack(fill="x")

        self._threshold = tk.IntVar(value=self.cfg.reader.threshold)
        self._invert = tk.BooleanVar(value=self.cfg.reader.invert)
        self._use_colour = tk.BooleanVar(value=self.cfg.reader.ink_color is not None)
        self._tolerance = tk.IntVar(value=self.cfg.reader.color_tolerance)

        ttk.Label(row, text="亮度門檻").pack(side="left")
        ttk.Scale(
            row,
            from_=0,
            to=255,
            variable=self._threshold,
            command=lambda _v: self._refresh_threshold_preview(),
            length=220,
        ).pack(side="left", padx=(6, 4))
        self.threshold_value = ttk.Label(row, width=4, text=str(self._threshold.get()))
        self.threshold_value.pack(side="left")

        ttk.Checkbutton(
            row,
            text="暗字亮底",
            variable=self._invert,
            command=self._refresh_threshold_preview,
        ).pack(side="left", padx=(14, 0))

        ttk.Checkbutton(
            row,
            text="使用色鍵",
            variable=self._use_colour,
            command=self._refresh_threshold_preview,
        ).pack(side="left", padx=(14, 0))

        ttk.Label(row, text="容許色距").pack(side="left", padx=(10, 0))
        ttk.Scale(
            row,
            from_=10,
            to=200,
            variable=self._tolerance,
            command=lambda _v: self._refresh_threshold_preview(),
            length=140,
        ).pack(side="left", padx=(6, 0))

        buttons = ttk.Frame(self.controls)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="自動建議", command=self._auto_threshold).pack(side="left")
        ttk.Button(buttons, text="重新擷取 ROI", command=self._refresh_threshold_preview).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="上一步", command=self._goto_roi_step).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(buttons, text="下一步", command=self._goto_glyph_step).pack(side="right")

        self._refresh_threshold_preview()

    def _capture_exp_roi(self) -> np.ndarray | None:
        if self.capturer is None:
            return None
        try:
            return self.capturer.capture_roi(self.cfg.reader.exp_roi).pixels
        except CaptureError as exc:
            self._set_status(f"擷取 ROI 失敗：{exc}", error=True)
            return None

    def _current_reader_settings(self) -> dict:
        return {
            "threshold": int(self._threshold.get()),
            "invert": bool(self._invert.get()),
            "ink_color": self.cfg.reader.ink_color if self._use_colour.get() else None,
            "color_tolerance": int(self._tolerance.get()),
        }

    def _refresh_threshold_preview(self) -> None:
        self.threshold_value.config(text=str(self._threshold.get()))
        pixels = self._capture_exp_roi()
        if pixels is None:
            return
        self._roi_pixels = pixels

        settings = self._current_reader_settings()
        mask = binarize(pixels, **settings)

        rgb = pngio.scale_nearest(pngio.bgra_to_rgb(pixels), ROI_ZOOM)
        self._roi_photo = tk.PhotoImage(data=pngio.to_tk_data(rgb))
        self.roi_canvas.delete("all")
        self.roi_canvas.config(width=rgb.shape[1], height=rgb.shape[0])
        self.roi_canvas.create_image(0, 0, anchor="nw", image=self._roi_photo)

        mask_rgb = pngio.scale_nearest(pngio.mask_to_rgb(mask), ROI_ZOOM)
        self._mask_photo = tk.PhotoImage(data=pngio.to_tk_data(mask_rgb))
        self.mask_canvas.delete("all")
        self.mask_canvas.config(width=mask_rgb.shape[1], height=mask_rgb.shape[0])
        self.mask_canvas.create_image(0, 0, anchor="nw", image=self._mask_photo)

        boxes = segment(
            mask,
            min_width=self.cfg.reader.min_glyph_width,
            min_height=self.cfg.reader.min_glyph_height,
            split_wide=self.cfg.reader.split_wide_glyphs,
            expected_width=self.templates.median_width(),
        )
        widths = ", ".join(f"{b.width}x{b.height}" for b in boxes)
        self.segment_label.config(
            text=f"切出 {len(boxes)} 個字元：{widths or '（無）'}"
        )
        if boxes:
            self._set_status(
                f"切出 {len(boxes)} 個字元。數字看起來分開且完整就可以下一步。"
            )

    def _pick_ink_colour(self, event) -> None:
        pixels = getattr(self, "_roi_pixels", None)
        if pixels is None:
            return
        x = int(event.x // ROI_ZOOM)
        y = int(event.y // ROI_ZOOM)
        if not (0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]):
            return
        bgr = [int(v) for v in pixels[y, x, :3]]
        self.cfg.reader.ink_color = bgr
        self._use_colour.set(True)
        self._set_status(f"已取色 BGR={bgr}，改用色鍵判斷")
        self._refresh_threshold_preview()

    def _auto_threshold(self) -> None:
        pixels = self._capture_exp_roi()
        if pixels is None:
            return
        threshold, invert = suggest_binarization(pixels)
        self._threshold.set(threshold)
        self._invert.set(invert)
        self._use_colour.set(False)
        self._set_status(f"自動建議：門檻 {threshold}、{'暗字亮底' if invert else '亮字暗底'}")
        self._refresh_threshold_preview()

    def _commit_reader_settings(self) -> None:
        settings = self._current_reader_settings()
        self.cfg.reader.threshold = settings["threshold"]
        self.cfg.reader.invert = settings["invert"]
        self.cfg.reader.color_tolerance = settings["color_tolerance"]
        if not self._use_colour.get():
            self.cfg.reader.ink_color = None

    # ------------------------------------------------------------------ #
    # 步驟 4：標字元
    # ------------------------------------------------------------------ #

    def _goto_glyph_step(self) -> None:
        self._commit_reader_settings()
        self._reset_panels(
            "步驟 4 / 5 — 標記字元",
            "程式會持續讀取狀態列，把「沒見過的字形」排進佇列。"
            "看著放大的圖輸入它是什麼字（數字按數字鍵；小數點、方括號、百分比符號也要標），"
            "按 Enter 送出。\n"
            "如果圖上明顯是兩個字黏在一起（最常見的是小數點貼著後面的數字，例如 .3），"
            "就直接把兩個字都打進去 —— 支援多字元標記，比硬去切開它可靠得多。\n"
            "請讓角色繼續打怪 —— 經驗值數字一直在變，10 個數字很快就會全部出現。"
            "0-9 全部收齊就可以進入下一步。",
        )

        top = ttk.Frame(self.body)
        top.pack(fill="both", expand=True)

        glyph_panel = ttk.LabelFrame(top, text="待標記字形")
        glyph_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.glyph_canvas = tk.Canvas(glyph_panel, background="#111", width=200, height=180)
        self.glyph_canvas.pack(padx=6, pady=6)

        entry_row = ttk.Frame(glyph_panel)
        entry_row.pack(pady=(0, 8))
        ttk.Label(entry_row, text="這是：").pack(side="left")
        self.glyph_entry = ttk.Entry(entry_row, width=6, font=("Segoe UI", 16))
        self.glyph_entry.pack(side="left", padx=6)
        self.glyph_entry.bind("<Return>", lambda _e: self._label_glyph())
        self.glyph_entry.focus_set()
        ttk.Button(entry_row, text="送出", command=self._label_glyph).pack(side="left")
        ttk.Button(entry_row, text="跳過", command=self._skip_glyph).pack(
            side="left", padx=(6, 0)
        )

        progress_panel = ttk.LabelFrame(top, text="收集進度")
        progress_panel.pack(side="left", fill="both", expand=True)
        self.progress_label = ttk.Label(
            progress_panel, text="", font=("Consolas", 12), justify="left"
        )
        self.progress_label.pack(anchor="nw", padx=10, pady=10)
        self.live_label = ttk.Label(
            progress_panel, text="", font=("Consolas", 11), justify="left"
        )
        self.live_label.pack(anchor="nw", padx=10, pady=(0, 10))

        ttk.Button(self.controls, text="清空所有模板", command=self._clear_templates).pack(
            side="left"
        )
        ttk.Button(self.controls, text="上一步", command=self._goto_threshold_step).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(self.controls, text="下一步", command=self._goto_verify_step).pack(
            side="right"
        )

        self._harvest_queue.clear()
        self._update_progress()
        self._harvest_tick()

    def _harvest_tick(self) -> None:
        """抓一次畫面，把無法辨識的字形排進佇列。"""
        pixels = self._capture_exp_roi()
        if pixels is not None:
            mask = binarize(
                pixels,
                threshold=self.cfg.reader.threshold,
                invert=self.cfg.reader.invert,
                ink_color=self.cfg.reader.ink_color,
                color_tolerance=self.cfg.reader.color_tolerance,
            )
            boxes = segment(
                mask,
                min_width=self.cfg.reader.min_glyph_width,
                min_height=self.cfg.reader.min_glyph_height,
                split_wide=self.cfg.reader.split_wide_glyphs,
                expected_width=self.templates.median_width(),
            )
            recognised = []
            for box in boxes:
                glyph = box.extract(mask)
                match = self.templates.match(
                    glyph,
                    min_score=self.cfg.reader.match_min_score,
                    min_margin=self.cfg.reader.match_min_margin,
                )
                if match.char:
                    recognised.append(match.char)
                else:
                    recognised.append("?")
                    self._enqueue_glyph(glyph)
            self.live_label.config(
                text=f"目前讀到：{''.join(recognised) or '（無）'}\n佇列：{len(self._harvest_queue)} 個待標"
            )

        self._show_next_glyph()
        self._harvest_job = self.after(HARVEST_INTERVAL_MS, self._harvest_tick)

    def _enqueue_glyph(self, glyph: np.ndarray) -> None:
        from ..vision.segment import normalize_glyph

        normalized = normalize_glyph(glyph)
        if normalized.size == 0:
            return
        if len(self._harvest_queue) >= 40:
            return
        for existing in self._harvest_queue:
            if existing.shape == normalized.shape and np.array_equal(existing, normalized):
                return
        self._harvest_queue.append(normalized)

    def _show_next_glyph(self) -> None:
        self.glyph_canvas.delete("all")
        if not self._harvest_queue:
            self.glyph_canvas.create_text(
                100, 90, text="（沒有待標記的字形）", fill="#888"
            )
            return
        glyph = self._harvest_queue[0]
        rgb = pngio.scale_nearest(pngio.mask_to_rgb(glyph), GLYPH_ZOOM)
        self._glyph_photo = tk.PhotoImage(data=pngio.to_tk_data(rgb))
        self.glyph_canvas.create_image(6, 6, anchor="nw", image=self._glyph_photo)

    def _label_glyph(self) -> None:
        if not self._harvest_queue:
            return
        text = self.glyph_entry.get().strip()
        self.glyph_entry.delete(0, tk.END)
        if not 1 <= len(text) <= 3:
            self._set_status("請輸入 1-3 個字元", error=True)
            return
        if any(c not in DIGITS + SYMBOLS for c in text):
            self._set_status(
                f"只支援 {DIGITS} 與符號 {SYMBOLS}（其他字元對計算沒有用）", error=True
            )
            return
        glyph = self._harvest_queue.pop(0)
        added = self.templates.add(text, glyph)
        self._set_status(
            f"已加入 {text!r}" if added else f"{text!r} 已有相同樣本，略過"
        )
        self._update_progress()
        self._show_next_glyph()

    def _skip_glyph(self) -> None:
        if self._harvest_queue:
            self._harvest_queue.pop(0)
            self.glyph_entry.delete(0, tk.END)
            self._show_next_glyph()

    def _clear_templates(self) -> None:
        if messagebox.askyesno("確認", "要清空所有已標記的字元模板嗎？"):
            self.templates.clear()
            self._harvest_queue.clear()
            self._update_progress()
            self._show_next_glyph()

    def _update_progress(self) -> None:
        collected: set[str] = set()
        for label in self.templates.chars():
            collected.update(label)   # 多字元標籤要攤平成單一字元來算進度
        digit_line = " ".join(
            f"{d}{'✓' if d in collected else '·'}" for d in DIGITS
        )
        symbol_line = " ".join(
            f"{s}{'✓' if s in collected else '·'}" for s in SYMBOLS
        )
        missing = [d for d in DIGITS if d not in collected]
        summary = "數字全部收齊" if not missing else f"還缺數字：{''.join(missing)}"
        self.progress_label.config(
            text=f"數字  {digit_line}\n符號  {symbol_line}\n\n{summary}\n"
            f"模板總數：{len(self.templates)}"
        )

    # ------------------------------------------------------------------ #
    # 步驟 5：驗證
    # ------------------------------------------------------------------ #

    def _goto_verify_step(self) -> None:
        missing = set(DIGITS) - set("".join(self.templates.chars()))
        if missing:
            if not messagebox.askyesno(
                "數字還沒收齊",
                f"還缺 {''.join(sorted(missing))}。缺少的數字一出現就會辨識失敗，"
                "確定要繼續嗎？",
            ):
                return

        self._reset_panels(
            "步驟 5 / 5 — 驗證",
            "確認下面讀到的數值跟遊戲畫面一致。\n"
            "最重要的一項是「推導所需經驗」：有值代表 ROI 同時包含絕對經驗值與百分比，"
            "升級、死亡都能正確處理，而且不需要任何內建經驗表。",
        )

        self.verify_label = ttk.Label(
            self.body, text="讀取中…", font=("Consolas", 12), justify="left"
        )
        self.verify_label.pack(anchor="nw", pady=10)

        ttk.Button(self.controls, text="回去補標字元", command=self._goto_glyph_step).pack(
            side="left"
        )
        ttk.Button(self.controls, text="儲存並結束", command=self._save_and_exit).pack(
            side="right"
        )
        self._verify_tick()

    def _verify_tick(self) -> None:
        lines: list[str] = []
        pixels = self._capture_exp_roi()
        if pixels is None:
            lines.append("擷取失敗")
        else:
            mask = binarize(
                pixels,
                threshold=self.cfg.reader.threshold,
                invert=self.cfg.reader.invert,
                ink_color=self.cfg.reader.ink_color,
                color_tolerance=self.cfg.reader.color_tolerance,
            )
            boxes = segment(
                mask,
                min_width=self.cfg.reader.min_glyph_width,
                min_height=self.cfg.reader.min_glyph_height,
                split_wide=self.cfg.reader.split_wide_glyphs,
                expected_width=self.templates.median_width(),
            )
            chars = []
            scores = []
            for box in boxes:
                match = self.templates.match(
                    box.extract(mask),
                    min_score=self.cfg.reader.match_min_score,
                    min_margin=self.cfg.reader.match_min_margin,
                )
                chars.append(match.char or "?")
                scores.append(match.score)
            text = "".join(chars)
            exp_abs, exp_pct = parse_exp_text(text)
            lines.append(f"辨識字串     : {text}")
            lines.append(f"最低相似度   : {min(scores):.2f}" if scores else "最低相似度   : --")
            lines.append(f"絕對經驗值   : {exp_abs if exp_abs is not None else '讀不到'}")
            lines.append(f"百分比       : {exp_pct if exp_pct is not None else '讀不到'}")
            if exp_abs is not None and exp_pct is not None and exp_pct >= 0.5:
                need = int(round(100.0 * exp_abs / exp_pct))
                lines.append(f"推導所需經驗 : {need:,}  （很好，不需要內建經驗表）")
            else:
                lines.append("推導所需經驗 : 無法推導 —— 請把百分比也框進 ROI")
        lines.append("等級         : 不用校準，追蹤時會從狀態列的橘色方塊自動讀")

        self.verify_label.config(text="\n".join(lines))
        self._verify_job = self.after(900, self._verify_tick)

    # ------------------------------------------------------------------ #

    def _save_and_exit(self) -> None:
        self._commit_reader_settings()
        self.templates.meta = {
            "threshold": self.cfg.reader.threshold,
            "invert": self.cfg.reader.invert,
            "ink_color": self.cfg.reader.ink_color,
            "color_tolerance": self.cfg.reader.color_tolerance,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.templates.save(self.templates_path)
        path = self.cfg.save()
        messagebox.showinfo(
            "已儲存",
            f"設定：{path}\n模板：{self.templates_path}\n\n"
            "接下來執行：\n  python -m mapleexp track --label 地圖名稱",
        )
        self._on_close()

    def _on_close(self) -> None:
        self._cancel_jobs()
        if self.capturer is not None:
            self.capturer.close()
            self.capturer = None
        self.destroy()


def run_calibration(cfg: Config | None = None) -> None:
    cfg = cfg or Config.load()
    templates_path = config_module.templates_dir() / "digits.json"
    CalibrationApp(cfg, templates_path).mainloop()
