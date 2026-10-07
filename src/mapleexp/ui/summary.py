"""結束或重設前的確認視窗。

顯示這一場的成果，並且**所有名稱都可以當場改** —— 地圖名的字太小，OCR 常常
讀不出來（角色名倒是很準）。所以設計上不依賴 OCR 的正確性：

* 真正用來區分地圖的是那塊像素的**指紋**，跟文字無關，所以改名字不會把統計弄亂。
* OCR 讀到什麼就先填什麼，讀不到就留空，使用者補一次，以後那張地圖都認得。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..core.stats import format_duration, format_exp, format_rate

BG = "#1b1d2b"
CARD = "#262a42"
FG = "#f2f3fa"
MUTED = "#9094b8"
MINT = "#7ee0b8"


class SummaryDialog(tk.Toplevel):
    """回傳 ``result``：True = 儲存，False = 不儲存，None = 取消。"""

    def __init__(self, parent, session, title: str = "本場練功紀錄") -> None:
        super().__init__(parent)
        self.session = session
        self.result: bool | None = None
        self._map_entries: list[tuple[str, tk.StringVar]] = []
        # 按了垃圾桶的段。先只在畫面上拿掉，按「儲存」才真的刪 —— 取消就什麼都沒發生。
        self._removed: list = []

        self.title(title)
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.update_idletasks()
        self._centre_on(parent)

    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Sum.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Sum.TLabel", background=BG, foreground=FG,
                        font=("Microsoft JhengHei UI", 10))
        style.configure("Dim.TLabel", background=BG, foreground=MUTED,
                        font=("Microsoft JhengHei UI", 9))
        style.configure("Head.TLabel", background=BG, foreground=MINT,
                        font=("Microsoft JhengHei UI", 11, "bold"))

        root = ttk.Frame(self, style="Sum.TFrame", padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="這一場要留下來嗎？", style="Head.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
        )

        ttk.Label(root, text="角色", style="Dim.TLabel").grid(row=1, column=0, sticky="w")
        self._character = tk.StringVar(value=self.session.character)
        ttk.Entry(root, textvariable=self._character, width=18).grid(
            row=1, column=1, sticky="w", padx=(6, 18)
        )
        ttk.Label(root, text="職業", style="Dim.TLabel").grid(row=1, column=2, sticky="w")
        self._job = tk.StringVar(value=self.session.job)
        ttk.Entry(root, textvariable=self._job, width=12).grid(
            row=1, column=3, sticky="w", padx=(6, 0)
        )

        ttk.Separator(root, orient="horizontal").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=12
        )

        self._table = ttk.Frame(root, style="Sum.TFrame")
        self._table.grid(row=3, column=0, columnspan=4, sticky="ew")
        self._hint = ttk.Label(
            root,
            text="紅色的欄位是自動辨識不出來的地圖名稱，填一次之後就會記住。",
            style="Dim.TLabel",
        )
        self._fill_table()

        buttons = ttk.Frame(root, style="Sum.TFrame")
        buttons.grid(row=5, column=0, columnspan=4, sticky="e", pady=(18, 0))
        ttk.Button(buttons, text="不要儲存", command=self._discard).pack(side="left")
        ttk.Button(buttons, text="儲存", command=self._save).pack(side="left", padx=(10, 0))

    def _fill_table(self) -> None:
        """（重新）畫表格。刪掉一列之後整張重畫最簡單，也不會留下對不齊的格子。"""
        table = self._table
        for child in table.winfo_children():
            child.destroy()
        self._map_entries.clear()

        titles = ("地圖名稱", "角色", "累計經驗值", "練了", "平均速度", "峰值", "活躍時間", "")
        for index, title in enumerate(titles):
            ttk.Label(table, text=title, style="Dim.TLabel").grid(
                row=0, column=index, sticky="w", padx=(0, 14), pady=(0, 6)
            )

        rows = self._rows()
        if not rows:
            ttk.Label(
                table, text="（這一場沒有要保留的紀錄）", style="Dim.TLabel"
            ).grid(row=1, column=0, columnspan=len(titles), sticky="w")
        for row_index, segment in enumerate(rows, start=1):
            variable = tk.StringVar(value=segment["name"])
            entry = ttk.Entry(table, textvariable=variable, width=18)
            entry.grid(row=row_index, column=0, sticky="w", padx=(0, 14), pady=2)
            if not segment["name"]:
                entry.configure(foreground="#a05050")
            self._map_entries.append((segment["map_id"], variable))

            pct = segment["pct"]
            cells = (
                segment["character"] or "-",
                format_exp(segment["exp"]),
                f"{pct:+.2f}%" if pct is not None else "--",
                format_rate(segment["rate"]),
                format_rate(segment["peak"]),
                format_duration(segment["active"]),
            )
            for column, text in enumerate(cells, start=1):
                ttk.Label(table, text=text, style="Sum.TLabel").grid(
                    row=row_index, column=column, sticky="e", padx=(0, 14)
                )
            self._trash_button(
                table, lambda seg=segment["segment"]: self._remove(seg)
            ).grid(row=row_index, column=len(cells) + 1, sticky="e")

        if any(not s["name"] for s in rows):
            self._hint.grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))
        else:
            self._hint.grid_remove()

    def _trash_button(self, parent, command) -> tk.Canvas:
        """垃圾桶圖示，自己畫的（字型裡的 🗑 在不同機器上長得不一樣、也對不齊）。"""
        size = 22
        canvas = tk.Canvas(parent, width=size, height=size, bg=BG,
                           highlightthickness=0, bd=0, cursor="hand2")
        colour = MUTED
        canvas.create_line(4, 6, 18, 6, fill=colour, width=2)           # 蓋子
        canvas.create_line(9, 4, 13, 4, fill=colour, width=2)           # 提把
        canvas.create_polygon(6, 8, 16, 8, 15, 19, 7, 19, outline=colour,
                              fill="", width=2)                        # 桶身
        canvas.create_line(9, 10, 9, 17, fill=colour, width=1)
        canvas.create_line(13, 10, 13, 17, fill=colour, width=1)
        canvas.bind("<Button-1>", lambda _e: command())
        canvas.bind("<Enter>", lambda _e: self._recolour_trash(canvas, "#ff8b94"))
        canvas.bind("<Leave>", lambda _e: self._recolour_trash(canvas, colour))
        return canvas

    @staticmethod
    def _recolour_trash(canvas: tk.Canvas, colour: str) -> None:
        """線條的顏色是 ``fill``，多邊形的線條卻是 ``outline``（``fill`` 是內部填滿）——
        用 ``itemconfigure("all", fill=...)`` 會把桶身填成實心。"""
        for item in canvas.find_all():
            if canvas.type(item) == "polygon":
                canvas.itemconfigure(item, outline=colour, fill="")
            else:
                canvas.itemconfigure(item, fill=colour)

    def _remove(self, segment) -> None:
        self._removed.append(segment)
        self._fill_table()

    def _rows(self) -> list[dict]:
        rows = []
        for segment in self.session.segments:
            if segment.active_sec <= 0 and segment.exp == 0:
                continue
            if any(segment is removed for removed in self._removed):
                continue
            rows.append(
                {
                    "segment": segment,
                    "map_id": segment.map_id,
                    "name": segment.name or self.session.map_names.get(segment.map_id, ""),
                    "character": segment.character,
                    "exp": segment.exp,
                    "pct": segment.pct_gained,
                    "rate": segment.exp_per_hour,
                    "peak": segment.peak_exp_per_hour,
                    "active": segment.active_sec,
                }
            )
        return rows

    def _centre_on(self, parent) -> None:
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_rooty() + 40
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ #

    def _apply_edits(self) -> None:
        self.session.character = self._character.get().strip()
        self.session.job = self._job.get().strip()
        for map_id, variable in self._map_entries:
            name = variable.get().strip()
            self.session.map_names[map_id] = name
            for segment in self.session.segments:
                if segment.map_id == map_id:
                    segment.name = name

    def _save(self) -> None:
        self._apply_edits()
        for segment in self._removed:
            self.session.delete_segment(segment)
        self.result = True
        self.destroy()

    def _discard(self) -> None:
        self.result = False
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class NameDialog(tk.Toplevel):
    """改一個名字的小視窗。``result`` 是新名字，取消則是 None。"""

    def __init__(self, parent, label: str, current: str, hint: str = "") -> None:
        super().__init__(parent)
        self.result: str | None = None
        self.title(label)
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.attributes("-topmost", True)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Name.TFrame", background=BG)
        style.configure("Name.TLabel", background=BG, foreground=FG,
                        font=("Microsoft JhengHei UI", 10))
        style.configure("NameDim.TLabel", background=BG, foreground=MUTED,
                        font=("Microsoft JhengHei UI", 8))

        root = ttk.Frame(self, style="Name.TFrame", padding=14)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text=label, style="Name.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self._value = tk.StringVar(value=current)
        entry = ttk.Entry(root, textvariable=self._value, width=26,
                          font=("Microsoft JhengHei UI", 11))
        entry.grid(row=1, column=0, columnspan=2, sticky="ew")
        entry.focus_set()
        entry.select_range(0, "end")
        if hint:
            ttk.Label(root, text=hint, style="NameDim.TLabel").grid(
                row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
            )

        buttons = ttk.Frame(root, style="Name.TFrame")
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="取消", command=self._cancel).pack(side="left")
        ttk.Button(buttons, text="確定", command=self._ok).pack(side="left", padx=(8, 0))

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        self.grab_set()
        try:
            x = parent.winfo_rootx() + 20
            y = parent.winfo_rooty() + parent.winfo_height() + 8
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    def _ok(self) -> None:
        self.result = self._value.get().strip()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class SettingsDialog(tk.Toplevel):
    """浮窗的設定：取樣頻率、閒置門檻。``result`` 是「有沒有改到東西」。"""

    MIN_INTERVAL, MAX_INTERVAL = 0.2, 10.0
    MAX_IDLE = 3600.0

    def __init__(self, parent, session) -> None:
        super().__init__(parent)
        self.session = session
        self.result = False
        self.title("設定")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.attributes("-topmost", True)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Set.TFrame", background=BG)
        style.configure("Set.TLabel", background=BG, foreground=FG,
                        font=("Microsoft JhengHei UI", 10))
        style.configure("SetErr.TLabel", background=BG, foreground="#ff8b94",
                        font=("Microsoft JhengHei UI", 8))

        cfg = session.cfg.tracker
        self._interval = tk.StringVar(value=f"{cfg.sample_interval:g}")
        self._idle = tk.StringVar(value=f"{cfg.idle_pause_sec:g}")

        root = ttk.Frame(self, style="Set.TFrame", padding=14)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="取樣頻率（秒/張）", style="Set.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        entry = ttk.Entry(root, textvariable=self._interval, width=8,
                          font=("Microsoft JhengHei UI", 11))
        entry.grid(row=0, column=1, sticky="e", padx=(16, 0), pady=(0, 8))

        ttk.Label(root, text="經驗停滯幾秒後算閒置（0 = 不判定）", style="Set.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(root, textvariable=self._idle, width=8,
                  font=("Microsoft JhengHei UI", 11)).grid(
            row=1, column=1, sticky="e", padx=(16, 0)
        )

        self._error = ttk.Label(root, text="", style="SetErr.TLabel")
        self._error.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(root, style="Set.TFrame")
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self._cancel).pack(side="left")
        ttk.Button(buttons, text="確定", command=self._ok).pack(side="left", padx=(8, 0))

        entry.focus_set()
        entry.select_range(0, "end")
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        self.grab_set()
        try:
            x = parent.winfo_rootx() + 20
            y = parent.winfo_rooty() + parent.winfo_height() + 8
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    def _parse(self) -> tuple[float, float] | None:
        try:
            interval = float(self._interval.get().strip())
            idle = float(self._idle.get().strip())
        except ValueError:
            self._error.config(text="請填數字。")
            return None
        if not (self.MIN_INTERVAL <= interval <= self.MAX_INTERVAL):
            self._error.config(
                text=f"取樣頻率要在 {self.MIN_INTERVAL:g}～{self.MAX_INTERVAL:g} 秒之間。"
            )
            return None
        if not (0.0 <= idle <= self.MAX_IDLE):
            self._error.config(text=f"閒置門檻要在 0～{self.MAX_IDLE:g} 秒之間。")
            return None
        return interval, idle

    def _ok(self) -> None:
        parsed = self._parse()
        if parsed is None:
            return
        interval, idle = parsed
        cfg = self.session.cfg.tracker
        if interval != cfg.sample_interval:
            self.session.set_interval(interval)
            self.result = True
        if idle != cfg.idle_pause_sec:
            # Tracker 拿的是同一個 TrackerConfig 物件，改了立刻生效。
            cfg.idle_pause_sec = idle
            self.result = True
        self.destroy()

    def _cancel(self) -> None:
        self.result = False
        self.destroy()


def ask_settings(parent, session) -> bool:
    """開設定視窗；回傳有沒有改到東西。"""
    dialog = SettingsDialog(parent, session)
    parent.wait_window(dialog)
    return dialog.result


def ask_name(parent, label: str, current: str = "", hint: str = "") -> str | None:
    dialog = NameDialog(parent, label, current, hint)
    parent.wait_window(dialog)
    return dialog.result


def ask_save(parent, session, title: str = "本場練功紀錄") -> bool | None:
    dialog = SummaryDialog(parent, session, title)
    parent.wait_window(dialog)
    return dialog.result
