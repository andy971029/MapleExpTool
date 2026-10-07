"""置頂浮窗。

整個畫面用一塊 Canvas 畫出來，而不是堆一堆 Label：

* 圖示是**畫**出來的（暫停是兩條圓角棒、重設是一圈帶箭頭的弧線），不是拿
  ``❚❚`` ``↻`` 這種字元去湊 —— 字元會因為系統字型不同而長得很醜，大小也對不齊。
* 數字一律用完整數字加千分位，不用 ``k``／``M``。縮寫看起來精簡，但要比較
  兩個速率差多少時，腦袋得先換算一次。
* 角色名、職業、地圖名顯示的是 **Windows 內建 OCR 讀出來的文字**（職業與地圖名
  還會跟清單比對過）。讀不到就留空，結束時的視窗可以補填；那塊像素本身不留，
  只拿來算指紋判斷「有沒有變、要不要重跑 OCR」。

用 Tk 的 ``after`` 驅動取樣，所以 UI 不會被擷取卡住，也不需要另開執行緒
去跟 Tk 搶主迴圈（Tk 不是 thread-safe，多一條執行緒只會多一種崩潰方式）。
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from pathlib import Path

from .. import config as config_module
from .. import update
from ..core.session import TrackingSession
from ..core.stats import (
    format_duration,
    format_elapsed,
    format_exp,
    format_rate,
)
from ..core.tracker import ACTIVE, IDLE, PAUSED

# 柔和一點的配色：深靛藍底 + 薄荷／蜜桃／天藍的點綴。
BG = "#1b1d2b"
CARD = "#262a42"
TRACK = "#343a5c"
BORDER = "#5d66a3"
FG = "#f2f3fa"
MUTED = "#9094b8"
MINT = "#7ee0b8"
PEACH = "#ffc48c"
CORAL = "#ff8b94"
SKY = "#7ec5ff"
SKY_SOFT = "#2f4a6e"

STATE_STYLE = {
    ACTIVE: ("追蹤中", MINT),
    IDLE: ("閒置中", PEACH),
    PAUSED: ("無法讀取", CORAL),
    # 還沒拿到第一筆取樣。對使用者來說跟閒置沒有差別。
    "warmup": ("閒置中", PEACH),
}

WIDTH = 430
HEIGHT = 294
# 縮小模式只留標題那一排。
HEIGHT_COMPACT = 44
PAD = 16
RADIUS = 10

HISTORY = 90

APP_TITLE = "MS EXP TOOL"


class OverlayApp(tk.Tk):
    def __init__(
        self, session: TrackingSession, update_check: update.UpdateCheck | None = None
    ) -> None:
        super().__init__()
        self.session = session
        cfg = session.cfg
        # 啟動時在背景查的更新結果；下載好等著換檔的新版放在 pending_update。
        self._update_check = update_check
        self.pending_update: Path | None = None

        # 浮窗本身沒有標題列，但這兩行會影響 alt-tab 與「本場紀錄」那個對話框 ——
        # iconbitmap(default=...) 設的是整個程式的預設圖示。
        self.title(APP_TITLE)
        icon = config_module.icon_path()
        if icon.exists():
            try:
                self.iconbitmap(default=str(icon))
            except tk.TclError:
                pass

        self.overrideredirect(True)
        self.attributes("-topmost", cfg.ui.always_on_top)
        try:
            self.attributes("-alpha", max(0.3, min(1.0, cfg.ui.overlay_alpha)))
        except tk.TclError:
            pass
        self.configure(bg=BG)
        self._compact = bool(cfg.ui.overlay_compact)
        self.geometry(f"{WIDTH}x{self._height()}+{cfg.ui.overlay_x}+{cfg.ui.overlay_y}")

        self.canvas = tk.Canvas(
            self, width=WIDTH, height=self._height(), bg=BG, highlightthickness=0, bd=0
        )
        self.canvas.pack(fill="both", expand=True)

        self._paused_by_user = False
        self._last_snapshot = session.tracker.snapshot()
        self._report = None
        self._closing = False
        self._drag_offset = (0, 0)
        self._dragging = False
        self._history: deque[float] = deque(maxlen=HISTORY)
        self._hitboxes: list[tuple[tuple[int, int, int, int], object]] = []

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # 從最小化的主控台啟動時，行程的啟動狀態會傳染給第一個視窗 ——
        # 視窗會被建成最小化，於是 DWM 根本不合成它，畫面上什麼都看不到。
        # 明確叫它出來。
        self.deiconify()
        self.lift()
        self._show_in_taskbar()
        self.after(50, self._tick)
        if update_check is not None:
            self.after(1500, self._poll_update)

    def _show_in_taskbar(self) -> None:
        """讓浮窗在工作列有自己的按鈕。

        ``overrideredirect`` 的視窗 Tk 會加上 ``WS_EX_TOOLWINDOW``，Windows 對這種
        視窗的約定就是「不進工作列、不進 alt-tab」。把它換成 ``WS_EX_APPWINDOW``
        再重新顯示一次，工作列就會出現按鈕（標題列照樣沒有）。要改的是 Tk 外層那個
        真正的頂層視窗，不是 ``winfo_id()`` 回傳的子視窗。
        """
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_APPWINDOW = 0x00040000
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
        except OSError:
            return
        self.update_idletasks()
        hwnd = user32.GetParent(self.winfo_id())
        if not hwnd:
            return
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        )
        # 樣式要在視窗重新顯示時才會被工作列採用。
        self.withdraw()
        self.deiconify()
        self.overrideredirect(True)
        self.attributes("-topmost", self.session.cfg.ui.always_on_top)
        self.lift()

    # ------------------------------------------------------------------ #
    # 取樣迴圈
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self._paused_by_user:
            self.after(300, self._tick)
            return
        result = self.session.tick()
        rate = self._main_rate(result.snapshot)
        if rate is not None:
            self._history.append(rate)
        self._last_snapshot = result.snapshot
        self._render(result.snapshot)
        self.after(max(50, int(self.session.sleep_until_next_tick() * 1000)), self._tick)

    # ------------------------------------------------------------------ #
    # 自動更新
    # ------------------------------------------------------------------ #

    def _poll_update(self) -> None:
        """等背景查詢完成。查到新版才打擾使用者；沒網路、沒新版都安靜略過。"""
        check = self._update_check
        if check is None or self._closing:
            return
        if not check.done:
            self.after(500, self._poll_update)
            return
        release = check.release
        if release is None or release.version == self.session.cfg.skipped_update:
            return
        self._offer_update(release)

    def _offer_update(self, release: update.Release) -> None:
        from .update import UPDATE_NOW, UPDATE_SKIP, ask_update, run_download

        self._paused_by_user = True
        try:
            choice = ask_update(self, release)
        finally:
            self._paused_by_user = False
            self.session.sleep_until_next_tick()
        if choice == UPDATE_SKIP:
            self.session.cfg.skipped_update = release.version
            try:
                self.session.cfg.save()
            except OSError:
                pass
            return
        if choice != UPDATE_NOW:
            return
        # 跟按 × 一樣：這一場有東西就問要不要存，取消就不更新。
        if not self._confirm("更新並重新啟動"):
            return
        self._paused_by_user = True
        try:
            path, error = run_download(self, release)
        finally:
            self._paused_by_user = False
        if error or path is None:
            from tkinter import messagebox

            messagebox.showerror(
                APP_TITLE,
                f"{error or '下載失敗。'}\n\n可以到這裡手動下載：\n{release.page}",
                parent=self,
            )
            # 這一場已經結算過了，重新開一場繼續追蹤。
            self.session.restart()
            self._history.clear()
            return
        # 換檔要等這個行程把資料庫、擷取器都關掉才做 —— 交給 cmd_track 收尾。
        self.pending_update = path
        try:
            self.session.cfg.save()
        except OSError:
            pass
        self._closing = True
        self.destroy()

    def _main_rate(self, snapshot) -> float | None:
        window = self.session.cfg.tracker.eta_window
        rate = snapshot.rates.get(window)
        if rate is not None and rate.valid:
            return rate.exp_per_hour
        for key in sorted(snapshot.rates):
            if snapshot.rates[key].valid:
                return snapshot.rates[key].exp_per_hour
        return None

    # ------------------------------------------------------------------ #
    # 繪圖小工具
    # ------------------------------------------------------------------ #

    def _round_rect(self, x1, y1, x2, y2, radius=RADIUS, **kwargs):
        """圓角矩形。Tk 沒有內建，用 smooth 的多邊形兜出來。"""
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1,
            x2, y1 + radius, x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2, x1, y2,
            x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)

    def _icon_button(self, cx: int, cy: int, kind: str, command, colour: str) -> None:
        """自己畫的圖示按鈕：圓角方框當底，圖示畫在中間。

        框是必要的 —— 沒有框的話三個圖示看起來像裝飾而不是按鈕，而且暫停／播放
        切換時只有圖示變，眼睛很難注意到。
        """
        canvas = self.canvas
        self._round_rect(
            cx - 12, cy - 12, cx + 12, cy + 12, radius=7,
            fill=TRACK, outline=BORDER, width=1,
        )
        if kind == "pause":
            for dx in (-4, 2):
                self._round_rect(
                    cx + dx, cy - 6, cx + dx + 3, cy + 6, radius=2, fill=colour, outline=""
                )
        elif kind == "play":
            canvas.create_polygon(
                cx - 4, cy - 6, cx - 4, cy + 6, cx + 6, cy,
                fill=colour, outline="", smooth=False,
            )
        elif kind == "reset":
            canvas.create_arc(
                cx - 7, cy - 7, cx + 7, cy + 7,
                start=50, extent=280, style="arc", outline=colour, width=2,
            )
            canvas.create_polygon(
                cx + 2, cy - 9, cx + 9, cy - 6, cx + 3, cy - 2,
                fill=colour, outline="",
            )
        elif kind == "close":
            canvas.create_line(cx - 5, cy - 5, cx + 5, cy + 5, fill=colour, width=2)
            canvas.create_line(cx + 5, cy - 5, cx - 5, cy + 5, fill=colour, width=2)
        elif kind in ("collapse", "expand"):
            # 縮小／放大：一個往上或往下的 V 形。
            direction = -1 if kind == "collapse" else 1
            canvas.create_line(
                cx - 5, cy - 2 * direction, cx, cy + 3 * direction, cx + 5, cy - 2 * direction,
                fill=colour, width=2, joinstyle="round", capstyle="round",
            )
        elif kind == "report":
            # 長條圖：三根高矮不一的柱子。
            for dx, height in ((-6, 5), (-1, 9), (4, 7)):
                self._round_rect(
                    cx + dx, cy + 6 - height, cx + dx + 3, cy + 6, radius=1,
                    fill=colour, outline="",
                )
        elif kind == "gear":
            # 齒輪：一個環加八根短齒。
            import math

            canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5, outline=colour, width=2
            )
            for step in range(8):
                angle = math.radians(step * 45)
                dx, dy = math.cos(angle), math.sin(angle)
                canvas.create_line(
                    cx + dx * 5, cy + dy * 5, cx + dx * 8.5, cy + dy * 8.5,
                    fill=colour, width=2,
                )
        self._hitboxes.append(((cx - 13, cy - 13, cx + 13, cy + 13), command))

    def _outlined_text(self, x, y, text, fill, outline, font, anchor="center"):
        """帶描邊的文字。

        經驗條上的數字會同時壓在「已填的藍色」與「未填的底色」上，單一顏色
        一定有一半看不清楚。描邊最簡單也最可靠。
        """
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            self.canvas.create_text(
                x + dx, y + dy, text=text, fill=outline, font=font, anchor=anchor
            )
        self.canvas.create_text(x, y, text=text, fill=fill, font=font, anchor=anchor)

    # ------------------------------------------------------------------ #
    # 繪製
    # ------------------------------------------------------------------ #

    def _height(self) -> int:
        return HEIGHT_COMPACT if self._compact else HEIGHT

    def _render(self, snapshot) -> None:
        self.canvas.delete("all")
        self._hitboxes.clear()
        self.canvas.create_rectangle(0, 0, WIDTH, self._height(), fill=BG, outline="")
        if self._compact:
            self._draw_compact(snapshot)
            return
        self._draw_header(snapshot)
        self._draw_headline(snapshot)
        self._draw_exp_bar(snapshot)
        self._draw_sparkline()
        self._draw_session(snapshot)

    def _state_style(self, snapshot) -> tuple[str, str]:
        label, colour = STATE_STYLE.get(snapshot.state, (snapshot.state, MUTED))
        if self._paused_by_user:
            # 使用者按的暫停跟「讀不到畫面」的暫停是兩回事，狀態字要講清楚。
            label, colour = "已暫停（手動）", PEACH
        return label, colour

    def _draw_buttons(self) -> None:
        """標題列右側那一排按鈕，放大與縮小模式共用。"""
        self._icon_button(WIDTH - PAD - 12, 22, "close", self._quit, CORAL)
        self._icon_button(
            WIDTH - PAD - 46, 22,
            "expand" if self._compact else "collapse",
            self._toggle_compact, MUTED,
        )
        self._icon_button(WIDTH - PAD - 80, 22, "reset", self._reset, MUTED)
        self._icon_button(
            WIDTH - PAD - 114, 22,
            "play" if self._paused_by_user else "pause",
            self._toggle_pause, MINT if self._paused_by_user else FG,
        )
        self._icon_button(WIDTH - PAD - 148, 22, "report", self._open_report, MUTED)
        self._icon_button(WIDTH - PAD - 182, 22, "gear", self._open_settings, MUTED)

    # 最左邊那顆按鈕的左緣，縮小模式的文字不能越過這裡。
    BUTTONS_LEFT = WIDTH - PAD - 182 - 12

    def _draw_header(self, snapshot) -> None:
        label, colour = self._state_style(snapshot)
        self.canvas.create_oval(PAD, 17, PAD + 10, 27, fill=colour, outline="")
        self.canvas.create_text(
            PAD + 18, 22, text=label, anchor="w", fill=colour,
            font=("Microsoft JhengHei UI", 10, "bold"),
        )
        self._draw_buttons()

    def _draw_compact(self, snapshot) -> None:
        """縮小模式：狀態燈、EXP/h、升級剩餘，然後是同一排按鈕。"""
        _label, colour = self._state_style(snapshot)
        self.canvas.create_oval(PAD, 17, PAD + 10, 27, fill=colour, outline="")
        self.canvas.create_text(
            PAD + 18, 22, text=format_rate(self._main_rate(snapshot)), anchor="w",
            fill=FG, font=("Segoe UI Semibold", 14),
        )
        self.canvas.create_text(
            self.BUTTONS_LEFT - 10, 22, text=_format_eta_short(snapshot.eta_sec),
            anchor="e", fill=FG if snapshot.eta_sec else MUTED,
            font=("Segoe UI Semibold", 12),
        )
        self._draw_buttons()

    def _open_report(self) -> None:
        """報表視窗不是模態的：開著它浮窗照樣追蹤。"""
        from .report import open_report

        self._report = open_report(self, self.session.store, self._report)

    def _toggle_compact(self) -> None:
        self._compact = not self._compact
        self.session.cfg.ui.overlay_compact = self._compact
        self.geometry(f"{WIDTH}x{self._height()}")
        self.canvas.config(height=self._height())
        try:
            self.session.cfg.save()
        except OSError:
            pass
        self._render(self._last_snapshot)

    def _draw_headline(self, snapshot) -> None:
        canvas = self.canvas
        rate = self._main_rate(snapshot)
        canvas.create_text(
            PAD, 62, text=format_rate(rate), anchor="w", fill=FG,
            font=("Segoe UI Semibold", 21),
        )

        level = snapshot.level if snapshot.level is not None else self.session.level_reader.level
        canvas.create_text(
            WIDTH - PAD, 48, text=f"Lv {level}" if level else "Lv --",
            anchor="e", fill=PEACH, font=("Segoe UI Semibold", 14),
        )
        # 角色名與職業用 OCR 讀出來的文字。讀不到就留空 —— 結束時的視窗可以補填。
        canvas.create_text(
            WIDTH - PAD, 70, text=self.session.character or "（未命名角色）",
            anchor="e", fill=FG, font=("Microsoft JhengHei UI", 10),
        )
        if self.session.job:
            canvas.create_text(
                WIDTH - PAD, 87, text=self.session.job, anchor="e", fill=MUTED,
                font=("Microsoft JhengHei UI", 9),
            )

    def _draw_exp_bar(self, snapshot) -> None:
        canvas = self.canvas
        top, bottom = 98, 126
        left, right = PAD, WIDTH - PAD
        self._round_rect(left, top, right, bottom, fill=TRACK, outline="")

        pct = snapshot.exp_pct
        if pct is not None:
            span = right - left
            filled = left + int(span * max(0.0, min(100.0, pct)) / 100.0)
            if filled - left > RADIUS * 2:
                self._round_rect(left, top, filled, bottom, fill=SKY, outline="")
            elif filled > left:
                canvas.create_rectangle(left, top, filled, bottom, fill=SKY, outline="")

        # 分子/分母(%) 直接放在經驗條上 —— 要看的三個數字就在同一個地方。
        if snapshot.exp_abs is not None:
            text = f"{format_exp(snapshot.exp_abs)} / {format_exp(snapshot.need)}"
            if pct is not None:
                text += f" ({pct:.2f}%)"
            self._outlined_text(
                (left + right) // 2, (top + bottom) // 2, text,
                fill=FG, outline="#10182c", font=("Segoe UI Semibold", 10),
            )

        canvas.create_text(
            left, bottom + 16, text=f"升級剩餘時間 {format_duration(snapshot.eta_sec)}",
            anchor="w", fill=FG if snapshot.eta_sec else MUTED,
            font=("Microsoft JhengHei UI", 10),
        )
        canvas.create_text(
            right, bottom + 16, text=f"還差 {format_exp(snapshot.remaining)}",
            anchor="e", fill=MUTED, font=("Microsoft JhengHei UI", 9),
        )

    def _draw_sparkline(self) -> None:
        canvas = self.canvas
        left, right = PAD, WIDTH - PAD
        top, bottom = 152, 216
        self._round_rect(left, top, right, bottom, fill=CARD, outline="")
        canvas.create_text(
            left + 10, top + 12, text="效率走勢", anchor="w", fill=MUTED,
            font=("Microsoft JhengHei UI", 8),
        )

        values = list(self._history)
        if len(values) < 2:
            canvas.create_text(
                (left + right) // 2, (top + bottom) // 2, text="收集資料中…",
                fill=MUTED, font=("Microsoft JhengHei UI", 9),
            )
            return

        # 速率全是 0 也要照畫（一條貼底的線）。顯示「收集資料中」會讓人以為
        # 程式沒在跑，但其實它跑得好好的，只是你沒在打怪。
        peak = max(values)
        scale = peak or 1.0          # 畫圖要避免除以零，但顯示要講真話
        step = (right - left - 16) / max(1, len(values) - 1)
        points: list[float] = []
        for index, value in enumerate(values):
            x = left + 8 + index * step
            y = bottom - 8 - (value / scale) * (bottom - top - 26)
            points.extend((x, y))

        canvas.create_polygon(
            [left + 8, bottom - 6, *points, right - 8, bottom - 6],
            fill=SKY_SOFT, outline="",
        )
        canvas.create_line(points, fill=SKY, width=2, smooth=True)
        canvas.create_text(
            right - 10, top + 12, text=f"最高 {format_rate(peak)}", anchor="e",
            fill=MUTED, font=("Microsoft JhengHei UI", 8),
        )

    def _draw_session(self, snapshot) -> None:
        canvas = self.canvas
        top = 226
        map_id = self.session.current_map_id
        map_name = self.session.map_names.get(map_id, "")
        canvas.create_text(
            PAD, top, text="地圖", anchor="nw", fill=MUTED,
            font=("Microsoft JhengHei UI", 8),
        )
        # 地圖名稱可以直接點來改。OCR 讀不到時尤其需要 ——
        # 與其讓人去翻指令列，不如就在看得到的地方改。
        label = map_name or "（點這裡命名）"
        item = canvas.create_text(
            PAD, top + 13, text=label, anchor="nw",
            fill=FG if map_name else PEACH, font=("Microsoft JhengHei UI", 11),
        )
        if map_id:
            left, name_top, right, name_bottom = canvas.bbox(item)
            canvas.create_line(
                left, name_bottom + 1, right, name_bottom + 1,
                fill=MUTED if map_name else PEACH,
            )
            self._hitboxes.append(
                ((left - 4, name_top - 4, right + 10, name_bottom + 6), self._edit_map_name)
            )

        right_x = WIDTH - PAD
        canvas.create_text(
            right_x, top, text="目前累積經驗值", anchor="ne", fill=MUTED,
            font=("Microsoft JhengHei UI", 8),
        )
        canvas.create_text(
            right_x, top + 12, text=format_exp(snapshot.cum_net), anchor="ne",
            fill=FG, font=("Segoe UI Semibold", 13),
        )

        idle = f" · 閒置 {format_elapsed(snapshot.idle_sec)}" if snapshot.idle_sec > 1 else ""
        canvas.create_text(
            PAD, top + 46, text=f"活躍 {format_elapsed(snapshot.active_sec)}{idle}",
            anchor="w", fill=MUTED, font=("Microsoft JhengHei UI", 9),
        )

    # ------------------------------------------------------------------ #
    # 互動
    # ------------------------------------------------------------------ #

    def _on_click(self, event) -> None:
        for (left, top, right, bottom), command in self._hitboxes:
            if left <= event.x <= right and top <= event.y <= bottom:
                # 點在按鈕上就不要進入拖曳模式，否則手一抖整個視窗會跟著跑。
                self._dragging = False
                command()
                return
        self._dragging = True
        self._drag_offset = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _on_drag(self, event) -> None:
        if not self._dragging:
            return
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        self.geometry(f"+{x}+{y}")
        self.session.cfg.ui.overlay_x = x
        self.session.cfg.ui.overlay_y = y

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _open_settings(self) -> None:
        from .summary import ask_settings

        was_paused = self._paused_by_user
        self._paused_by_user = True
        try:
            changed = ask_settings(self, self.session)
        finally:
            self._paused_by_user = was_paused
            self.session.sleep_until_next_tick()
        if changed:
            try:
                self.session.cfg.save()
            except OSError:
                pass
        self._render(self._last_snapshot)

    def _edit_map_name(self) -> None:
        """點地圖名稱就能改。改完立刻存進資料庫，之後靠指紋自動對上。"""
        from .summary import ask_name

        map_id = self.session.current_map_id
        if not map_id:
            return
        current = self.session.map_names.get(map_id, "")
        was_paused = self._paused_by_user
        self._paused_by_user = True
        try:
            name = ask_name(
                self,
                "這張地圖叫什麼？",
                current,
                hint="改過之後，以後進到同一張圖都會自動認出來。",
            )
        finally:
            self._paused_by_user = was_paused
            self.session.sleep_until_next_tick()
        if name is None:
            return
        self.session.rename_map(map_id, name)

    def _toggle_pause(self) -> None:
        self._paused_by_user = not self._paused_by_user
        if not self._paused_by_user:
            self.session.sleep_until_next_tick()
        # 暫停期間 _tick 不會重畫，按鈕要在這裡立刻換成播放圖示。
        self._render(self._last_snapshot)

    def _reset(self) -> None:
        if not self._confirm("重新開始一場"):
            return
        self.session.restart()
        self._history.clear()

    def _quit(self) -> None:
        if not self._confirm("結束追蹤"):
            return
        try:
            self.session.cfg.save()
        except OSError:
            pass
        self._closing = True
        self.destroy()

    def _confirm(self, title: str) -> bool:
        """問使用者這一場要不要留。回傳是否要繼續執行這個動作。

        一點經驗都沒拿到（開著程式但還沒開始打）就沒什麼好問的，直接丟掉。
        """
        from .summary import ask_save

        if self.session.tracker.snapshot().cum_gross <= 0:
            self.session.finish(save=False)
            return True

        self._paused_by_user = True
        try:
            answer = ask_save(self, self.session, title)
        finally:
            self._paused_by_user = False
            self.session.sleep_until_next_tick()
        if answer is None:
            return False
        self.session.finish(save=answer)
        return True


def _format_eta_short(seconds: float | None) -> str:
    """縮小模式的升級剩餘：``1h23m``，不到一小時就 ``23m``。沒資料是 ``--``。"""
    if seconds is None or seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, minutes = divmod(total // 60, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def run_overlay(
    session: TrackingSession, update_check: update.UpdateCheck | None = None
) -> Path | None:
    """跑浮窗直到使用者關掉。回傳下載好、等著換檔的新版路徑（沒有就是 None）。"""
    app = OverlayApp(session, update_check)
    app.mainloop()
    return app.pending_update
