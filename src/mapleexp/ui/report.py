"""練功紀錄報表視窗。

把資料庫裡所有**已儲存**的段（角色 × 地圖）列成一張表。正在進行中、還沒按儲存的
那一場不在這裡 —— 它還在記憶體裡，結束視窗才會看到。

* 點欄位標題排序，再點一次反向。
* 角色／職業／地圖的標題右邊有個漏斗，點它會列出現有資料裡的所有值讓你勾選；
  開始時間的漏斗則是選日期區間。有篩選生效的漏斗會變成薄荷綠。

不是模態視窗：開著它浮窗照樣追蹤。只讀資料庫，跟追蹤的寫入走同一條連線、同一個
執行緒，不會互相卡。
"""

from __future__ import annotations

import datetime as _dt
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from ..core.stats import format_duration, format_exp, format_rate
from ..vision import pngio

BG = "#1b1d2b"
CARD = "#262a42"
TRACK = "#343a5c"
FG = "#f2f3fa"
MUTED = "#9094b8"
MINT = "#7ee0b8"
CORAL = "#ff8b94"

# (欄位鍵, 標題, 型別, 寬度, 對齊, 篩選方式)。
# 型別決定排序（文字 vs 數字）；篩選方式：``set`` 勾選清單、``date`` 日期區間、None 不能篩。
COLUMNS: tuple[tuple[str, str, str, int, str, str | None], ...] = (
    ("started", "開始時間", "text", 168, "w", "date"),
    ("character", "角色", "text", 130, "w", "set"),
    ("job", "職業", "text", 80, "w", "set"),
    ("map", "地圖", "text", 175, "w", "set"),
    ("exp", "經驗", "num", 95, "e", None),
    ("pct", "練了 %", "num", 75, "e", None),
    ("active", "活躍", "num", 75, "e", None),
    ("rate", "平均 /h", "num", 95, "e", None),
    ("peak", "峰值 /h", "num", 95, "e", None),
    ("deaths", "死亡", "num", 55, "e", None),
    ("levelups", "升等", "num", 55, "e", None),
)
_KINDS = {c[0]: c[2] for c in COLUMNS}
_WIDTHS = {c[0]: c[3] for c in COLUMNS}
_FILTERS = {c[0]: c[5] for c in COLUMNS}

# 標題右側這麼多像素算漏斗的點擊範圍。
ICON_ZONE = 26
HEADING_HEIGHT = 28


def _row_from_record(record: dict) -> dict:
    """把 Store.list_segments 的一筆轉成「原始值 + 顯示字串」。

    原始值拿來排序與篩選，顯示字串拿來畫表格。
    """
    active = float(record.get("active_sec") or 0.0)
    exp = int(record.get("exp") or 0)
    rate = exp / (active / 3600.0) if active > 0 else None
    raw = {
        "started": float(record.get("started_at") or 0.0),
        "character": record.get("character") or "",
        "job": record.get("job") or "",
        "map": record.get("map_name") or f"(未命名 {str(record.get('map_id', ''))[:6]})",
        "exp": exp,
        "pct": record.get("pct_gained"),
        "active": active,
        "rate": rate,
        "peak": record.get("peak_exp_per_hour"),
        "deaths": int(record.get("deaths") or 0),
        "levelups": int(record.get("levelups") or 0),
    }
    pct = raw["pct"]
    shown = {
        "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(raw["started"])),
        "character": raw["character"] or "-",
        "job": raw["job"] or "-",
        "map": raw["map"],
        "exp": format_exp(exp),
        "pct": f"{pct:+.2f}%" if pct is not None else "--",
        "active": format_duration(active),
        "rate": format_rate(rate),
        "peak": format_rate(raw["peak"]),
        "deaths": str(raw["deaths"]),
        "levelups": str(raw["levelups"]),
    }
    return {"raw": raw, "shown": shown}


def _sort_key(row: dict, key: str, kind: str):
    value = row["raw"][key]
    if kind == "num":
        # None（沒有峰值之類）一律排最後，升降都一樣。
        return (value is None, float(value) if value is not None else 0.0)
    return (False, str(value).casefold())


def _passes(
    row: dict,
    selections: dict[str, set[str] | None],
    date_range: tuple[float | None, float | None],
) -> bool:
    """一列過不過得了目前的篩選。

    ``selections`` 的值是「被勾選的顯示文字」集合；None 代表那一欄沒有篩選。
    ``date_range`` 是 (起, 迄) 的時間戳；迄已經含到那一天結束。
    """
    for key, chosen in selections.items():
        if chosen is not None and row["shown"][key] not in chosen:
            return False
    start, end = date_range
    started = row["raw"]["started"]
    if start is not None and started < start:
        return False
    if end is not None and started >= end:
        return False
    return True


def _parse_date(text: str) -> float | None:
    """``2026-10-07``／``2026/10/7``／``20261007`` -> 當天 00:00 的時間戳；空字串 -> None。"""
    text = (text or "").strip()
    if not text:
        return None
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            day = _dt.datetime.strptime(text, pattern)
        except ValueError:
            continue
        return time.mktime(day.timetuple())
    raise ValueError(text)


def _day_start(days_ago: int = 0) -> float:
    today = _dt.date.today() - _dt.timedelta(days=days_ago)
    return time.mktime(today.timetuple())


def _format_day(timestamp: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(timestamp)) if timestamp else ""


def _funnel_image(rgb: tuple[int, int, int], size: int = 14) -> tk.PhotoImage:
    """漏斗圖示：上面一個倒梯形、下面一根柄。畫成 PNG 餵給 Tk，背景透明。"""
    image = np.zeros((size, size, 4), dtype=np.uint8)
    for step, y in enumerate(range(2, 7)):
        left, right = 1 + step, size - 1 - step
        image[y, left:right, :3] = rgb
        image[y, left:right, 3] = 255
    image[7:12, 6:8, :3] = rgb
    image[7:12, 6:8, 3] = 255
    return tk.PhotoImage(data=pngio.to_tk_data(image))


def _hex_to_rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


# --------------------------------------------------------------------------- #
# 漏斗彈出視窗
# --------------------------------------------------------------------------- #


class _Popup(tk.Toplevel):
    """無標題列的小視窗，貼在標題下面；點外面或按 Esc 就套用並關閉。"""

    def __init__(self, parent, x: int, y: int) -> None:
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=TRACK)
        self.attributes("-topmost", True)
        self._anchor = (x, y)
        self.bind("<Escape>", lambda _e: self.close())
        self.bind("<Button-1>", self._maybe_close, add="+")

    def show(self) -> None:
        self.update_idletasks()
        x, y = self._anchor
        screen_w = self.winfo_screenwidth()
        width = self.winfo_reqwidth()
        self.geometry(f"+{max(0, min(x, screen_w - width))}+{y}")
        self.deiconify()
        self.grab_set()
        self.focus_set()

    def _maybe_close(self, event) -> None:
        inside = (
            0 <= event.x_root - self.winfo_rootx() < self.winfo_width()
            and 0 <= event.y_root - self.winfo_rooty() < self.winfo_height()
        )
        if not inside:
            self.close()

    def close(self) -> None:
        self.apply()
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    def apply(self) -> None:  # pragma: no cover - 子類別覆寫
        pass


class _SetFilterPopup(_Popup):
    """勾選清單：現有資料裡的所有值。"""

    MAX_VISIBLE = 14

    def __init__(self, parent, x, y, values: list[str], chosen: set[str] | None, on_apply) -> None:
        super().__init__(parent, x, y)
        self._on_apply = on_apply
        self._vars: dict[str, tk.BooleanVar] = {}

        body = tk.Frame(self, bg=CARD, bd=1, relief="solid", highlightthickness=0)
        body.pack(fill="both", expand=True)

        actions = tk.Frame(body, bg=CARD)
        actions.pack(fill="x", padx=6, pady=(6, 2))
        for text, value in (("全選", True), ("清除", False)):
            tk.Button(
                actions, text=text, command=lambda v=value: self._set_all(v),
                bg=TRACK, fg=FG, activebackground="#434b78", activeforeground=FG,
                relief="flat", padx=8, font=("Microsoft JhengHei UI", 9),
            ).pack(side="left", padx=(0, 4))

        # 值多的時候（地圖）要能捲。
        canvas = tk.Canvas(body, bg=CARD, highlightthickness=0, bd=0, width=220)
        inner = tk.Frame(canvas, bg=CARD)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        if len(values) > self.MAX_VISIBLE:
            scroll.pack(side="right", fill="y", pady=4)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        for value in values:
            variable = tk.BooleanVar(value=chosen is None or value in chosen)
            self._vars[value] = variable
            tk.Checkbutton(
                inner, text=value, variable=variable, anchor="w",
                bg=CARD, fg=FG, selectcolor=TRACK, activebackground=CARD,
                activeforeground=FG, font=("Microsoft JhengHei UI", 10),
            ).pack(fill="x")
        inner.update_idletasks()
        height = min(inner.winfo_reqheight(), 24 * self.MAX_VISIBLE)
        canvas.configure(
            width=max(220, inner.winfo_reqwidth()), height=height,
            scrollregion=(0, 0, inner.winfo_reqwidth(), inner.winfo_reqheight()),
        )
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"))

        tk.Button(
            body, text="確定", command=self.close,
            bg=TRACK, fg=FG, activebackground="#434b78", activeforeground=FG,
            relief="flat", padx=12, font=("Microsoft JhengHei UI", 9),
        ).pack(side="bottom", anchor="e", padx=6, pady=(2, 6))

    def _set_all(self, value: bool) -> None:
        for variable in self._vars.values():
            variable.set(value)

    def apply(self) -> None:
        chosen = {value for value, variable in self._vars.items() if variable.get()}
        # 全部勾起來就等於沒有篩選。
        self._on_apply(None if len(chosen) == len(self._vars) else chosen)

    def destroy(self) -> None:
        try:
            self.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        super().destroy()


class _DateFilterPopup(_Popup):
    """日期區間：起、迄各一格，加幾個常用的快速鍵。"""

    def __init__(self, parent, x, y, current: tuple[float | None, float | None], on_apply) -> None:
        super().__init__(parent, x, y)
        self._on_apply = on_apply
        start, end = current
        # 迄存的是「那一天結束」，顯示時退回那一天。
        self._start = tk.StringVar(value=_format_day(start))
        self._end = tk.StringVar(value=_format_day(end - 1) if end else "")

        body = tk.Frame(self, bg=CARD, bd=1, relief="solid", highlightthickness=0)
        body.pack(fill="both", expand=True)
        grid = tk.Frame(body, bg=CARD)
        grid.pack(padx=10, pady=(8, 4))
        for row, (label, variable) in enumerate((("起", self._start), ("迄", self._end))):
            tk.Label(grid, text=label, bg=CARD, fg=MUTED,
                     font=("Microsoft JhengHei UI", 9)).grid(row=row, column=0, sticky="w")
            tk.Entry(grid, textvariable=variable, width=12, bg=TRACK, fg=FG,
                     insertbackground=FG, relief="flat",
                     font=("Microsoft JhengHei UI", 10)).grid(
                row=row, column=1, padx=(6, 0), pady=2
            )
        tk.Label(grid, text="YYYY-MM-DD，留空代表不限", bg=CARD, fg=MUTED,
                 font=("Microsoft JhengHei UI", 8)).grid(row=2, column=0, columnspan=2, sticky="w")

        quick = tk.Frame(body, bg=CARD)
        quick.pack(fill="x", padx=10, pady=(2, 4))
        for text, days in (("今天", 0), ("近 7 天", 6), ("近 30 天", 29), ("全部", None)):
            tk.Button(
                quick, text=text, command=lambda d=days: self._quick(d),
                bg=TRACK, fg=FG, activebackground="#434b78", activeforeground=FG,
                relief="flat", padx=6, font=("Microsoft JhengHei UI", 9),
            ).pack(side="left", padx=(0, 4))

        self._error = tk.Label(body, text="", bg=CARD, fg=CORAL,
                               font=("Microsoft JhengHei UI", 8))
        self._error.pack(anchor="w", padx=10)
        tk.Button(
            body, text="確定", command=self.close,
            bg=TRACK, fg=FG, activebackground="#434b78", activeforeground=FG,
            relief="flat", padx=12, font=("Microsoft JhengHei UI", 9),
        ).pack(anchor="e", padx=10, pady=(0, 8))

    def _quick(self, days_ago: int | None) -> None:
        if days_ago is None:
            self._start.set("")
            self._end.set("")
        else:
            self._start.set(_format_day(_day_start(days_ago)))
            self._end.set(_format_day(_day_start(0)))

    def _parsed(self) -> tuple[float | None, float | None] | None:
        try:
            start = _parse_date(self._start.get())
            end = _parse_date(self._end.get())
        except ValueError as exc:
            self._error.config(text=f"看不懂的日期：{exc}")
            return None
        if end is not None:
            end += 86400.0           # 含到那一天結束
        if start is not None and end is not None and start >= end:
            self._error.config(text="起日不能晚於迄日")
            return None
        return start, end

    def close(self) -> None:
        if self._parsed() is None:
            return                   # 日期有問題就不關，讓人改
        super().close()

    def apply(self) -> None:
        parsed = self._parsed()
        if parsed is not None:
            self._on_apply(parsed)


# --------------------------------------------------------------------------- #
# 報表視窗
# --------------------------------------------------------------------------- #


class ReportWindow(tk.Toplevel):
    def __init__(self, parent, store) -> None:
        super().__init__(parent)
        self.title("練功紀錄")
        self.configure(bg=BG)
        self.attributes("-topmost", True)
        self._store = store
        self._rows: list[dict] = []
        # 勾選式篩選：欄位 -> 被勾的值（顯示文字）；None 代表沒篩。
        self._selections: dict[str, set[str] | None] = {
            key: None for key, kind in _FILTERS.items() if kind == "set"
        }
        self._date_range: tuple[float | None, float | None] = (None, None)
        self._sort_key: str = "started"
        self._sort_desc: bool = True
        self._icons = {
            "idle": _funnel_image(_hex_to_rgb(MUTED)),
            "active": _funnel_image(_hex_to_rgb(MINT)),
        }

        self._style()
        self._build()
        self.reload()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _e: self.destroy())
        try:
            x = parent.winfo_rootx()
            y = parent.winfo_rooty() + parent.winfo_height() + 8
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ #

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Rep.TFrame", background=BG)
        style.configure("Rep.TLabel", background=BG, foreground=MUTED,
                        font=("Microsoft JhengHei UI", 9))
        style.configure("Rep.Treeview", background=CARD, fieldbackground=CARD,
                        foreground=FG, rowheight=24, borderwidth=0,
                        font=("Microsoft JhengHei UI", 10))
        style.configure("Rep.Treeview.Heading", background=TRACK, foreground=FG,
                        relief="flat", padding=(4, 5),
                        font=("Microsoft JhengHei UI", 10, "bold"))
        style.map("Rep.Treeview.Heading", background=[("active", "#434b78")])
        style.map("Rep.Treeview", background=[("selected", "#3a5a8c")])

    def _build(self) -> None:
        root = ttk.Frame(self, style="Rep.TFrame", padding=12)
        root.pack(fill="both", expand=True)

        table = ttk.Frame(root, style="Rep.TFrame")
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            table, columns=[c[0] for c in COLUMNS], show="headings",
            style="Rep.Treeview", selectmode="browse",
        )
        for key, title, _kind, width, anchor, filter_kind in COLUMNS:
            self.tree.heading(key, text=title, anchor="center")
            if filter_kind:
                self.tree.heading(key, image=self._icons["idle"])
            self.tree.column(key, width=width, minwidth=40, anchor=anchor, stretch=False)
        # 標題的點擊自己處理：右邊那塊是漏斗，其餘是排序。
        self.tree.bind("<Button-1>", self._on_click)
        scroll_y = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll_x = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)

        footer = ttk.Frame(root, style="Rep.TFrame")
        footer.pack(fill="x", pady=(8, 0))
        self._summary = ttk.Label(footer, text="", style="Rep.TLabel")
        self._summary.pack(side="left")
        ttk.Button(footer, text="重新讀取", command=self.reload).pack(side="right")
        ttk.Button(footer, text="清除篩選", command=self._clear_filters).pack(
            side="right", padx=(0, 8)
        )
        self.minsize(sum(_WIDTHS.values()) + 60, 320)

    # ------------------------------------------------------------------ #

    def reload(self) -> None:
        records = self._store.list_segments() if self._store is not None else []
        self._rows = [_row_from_record(r) for r in records]
        self._refresh()

    def _clear_filters(self) -> None:
        for key in self._selections:
            self._selections[key] = None
        self._date_range = (None, None)
        self._refresh()

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) != "heading":
            return None
        column = self.tree.identify_column(event.x)
        if not column:
            return None
        key = COLUMNS[int(column[1:]) - 1][0]
        # 這一欄在畫面上的右緣：前面所有欄寬加總，扣掉水平捲動。
        offset = self.tree.xview()[0] * sum(_WIDTHS.values())
        right = sum(_WIDTHS[c[0]] for c in COLUMNS[: int(column[1:])]) - offset
        if _FILTERS[key] and event.x >= right - ICON_ZONE:
            self._open_filter(key, int(self.tree.winfo_rootx() + right))
        else:
            self._sort_by(key)
        return "break"

    def _open_filter(self, key: str, right_x: int) -> None:
        y = self.tree.winfo_rooty() + HEADING_HEIGHT
        if _FILTERS[key] == "date":
            popup = _DateFilterPopup(self, right_x - 260, y, self._date_range, self._set_date_range)
        else:
            values = sorted({row["shown"][key] for row in self._rows}, key=str.casefold)
            popup = _SetFilterPopup(
                self, right_x - 240, y, values, self._selections[key],
                lambda chosen, k=key: self._set_selection(k, chosen),
            )
        popup.show()

    def _set_selection(self, key: str, chosen: set[str] | None) -> None:
        self._selections[key] = chosen
        self._refresh()

    def _set_date_range(self, date_range: tuple[float | None, float | None]) -> None:
        self._date_range = date_range
        self._refresh()

    def _sort_by(self, key: str) -> None:
        if key == self._sort_key:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_key = key
            # 數字欄第一次點預設由大到小（通常是想看最高的），文字欄由小到大。
            self._sort_desc = _KINDS[key] == "num"
        self._refresh()

    def _visible_rows(self) -> list[dict]:
        rows = [
            row for row in self._rows
            if _passes(row, self._selections, self._date_range)
        ]
        kind = _KINDS[self._sort_key]
        rows.sort(key=lambda row: _sort_key(row, self._sort_key, kind), reverse=self._sort_desc)
        if self._sort_desc and kind == "num":
            # reverse 會把「None 排最後」翻成最前，再把它們移回去。
            rows = [r for r in rows if r["raw"][self._sort_key] is not None] + [
                r for r in rows if r["raw"][self._sort_key] is None
            ]
        return rows

    def _refresh(self) -> None:
        for key, title, _kind, _width, _anchor, filter_kind in COLUMNS:
            marker = ""
            if key == self._sort_key:
                marker = " ▼" if self._sort_desc else " ▲"
            self.tree.heading(key, text=title + marker)
            if filter_kind:
                active = (
                    self._date_range != (None, None) if filter_kind == "date"
                    else self._selections.get(key) is not None
                )
                self.tree.heading(key, image=self._icons["active" if active else "idle"])

        self.tree.delete(*self.tree.get_children())
        rows = self._visible_rows()
        for row in rows:
            self.tree.insert("", "end", values=[row["shown"][c[0]] for c in COLUMNS])

        exp = sum(r["raw"]["exp"] for r in rows)
        active = sum(r["raw"]["active"] for r in rows)
        rate = format_rate(exp / (active / 3600.0)) if active > 0 else "--"
        self._summary.config(
            text=f"{len(rows)} / {len(self._rows)} 筆 · 經驗 {format_exp(exp)} · "
                 f"活躍 {format_duration(active)} · 平均 {rate}"
        )


def open_report(parent, store, existing: ReportWindow | None = None) -> ReportWindow:
    """開報表視窗；已經開著就拉到前面並重新讀取。"""
    if existing is not None and existing.winfo_exists():
        existing.reload()
        existing.lift()
        existing.focus_force()
        return existing
    return ReportWindow(parent, store)
