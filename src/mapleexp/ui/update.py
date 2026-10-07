"""自動更新的兩個小視窗：「有新版本」的詢問，以及下載進度。

下載在背景執行緒做，視窗用 ``after`` 輪詢進度 —— Tk 不是 thread-safe，執行緒
只負責更新幾個數字，畫面一律由主迴圈畫。
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .. import __version__, update

BG = "#1b1d2b"
FG = "#f2f3fa"
MUTED = "#9094b8"
MINT = "#7ee0b8"

UPDATE_NOW = "now"
UPDATE_LATER = "later"
UPDATE_SKIP = "skip"

FONT = ("Microsoft JhengHei UI", 10)
FONT_SMALL = ("Microsoft JhengHei UI", 8)
FONT_TITLE = ("Microsoft JhengHei UI", 11, "bold")


def _style(widget: tk.Misc) -> None:
    style = ttk.Style(widget)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Upd.TFrame", background=BG)
    style.configure("Upd.TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("UpdTitle.TLabel", background=BG, foreground=MINT, font=FONT_TITLE)
    style.configure("UpdDim.TLabel", background=BG, foreground=MUTED, font=FONT_SMALL)
    style.configure("Upd.Horizontal.TProgressbar", troughcolor="#343a5c", background=MINT)


def _place_near(dialog: tk.Toplevel, parent) -> None:
    dialog.update_idletasks()
    try:
        x = parent.winfo_rootx() + 20
        y = parent.winfo_rooty() + parent.winfo_height() + 8
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    except tk.TclError:
        pass


class UpdateDialog(tk.Toplevel):
    """「有新版本 vX，要更新嗎？」``result`` 是 UPDATE_NOW / UPDATE_LATER / UPDATE_SKIP。"""

    def __init__(self, parent, release: update.Release) -> None:
        super().__init__(parent)
        self.result = UPDATE_LATER
        self.title("有新版本")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.attributes("-topmost", True)
        _style(self)

        root = ttk.Frame(self, style="Upd.TFrame", padding=16)
        root.pack(fill="both", expand=True)
        ttk.Label(
            root, text=f"MS EXP TOOL 有新版本 v{release.version}", style="UpdTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            root, text=f"目前是 v{__version__}。更新會下載新版、換掉這個檔案，然後重新開啟。",
            style="Upd.TLabel", wraplength=360, justify="left",
        ).pack(anchor="w", pady=(6, 0))
        if release.notes:
            notes = release.notes
            if len(notes) > 600:
                notes = notes[:600].rstrip() + "…"
            box = tk.Text(
                root, height=min(8, notes.count("\n") + 2), width=46, wrap="word",
                bg="#262a42", fg=FG, relief="flat", font=FONT_SMALL, padx=8, pady=6,
                highlightthickness=0,
            )
            box.insert("1.0", notes)
            box.configure(state="disabled")
            box.pack(fill="x", pady=(10, 0))
        ttk.Label(
            root, text="正在追蹤的這一場會先照一般流程問你要不要存。",
            style="UpdDim.TLabel",
        ).pack(anchor="w", pady=(10, 0))

        buttons = ttk.Frame(root, style="Upd.TFrame")
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="略過這版", command=lambda: self._pick(UPDATE_SKIP)).pack(
            side="left"
        )
        ttk.Button(buttons, text="稍後", command=lambda: self._pick(UPDATE_LATER)).pack(
            side="right"
        )
        now = ttk.Button(buttons, text="立即更新", command=lambda: self._pick(UPDATE_NOW))
        now.pack(side="right", padx=(0, 8))
        now.focus_set()

        self.bind("<Return>", lambda _e: self._pick(UPDATE_NOW))
        self.bind("<Escape>", lambda _e: self._pick(UPDATE_LATER))
        self.protocol("WM_DELETE_WINDOW", lambda: self._pick(UPDATE_LATER))
        _place_near(self, parent)
        self.grab_set()

    def _pick(self, choice: str) -> None:
        self.result = choice
        self.destroy()


class DownloadDialog(tk.Toplevel):
    """下載新版。成功時 ``path`` 是暫存檔；失敗時 ``error`` 是給使用者看的訊息。"""

    def __init__(self, parent, release: update.Release) -> None:
        super().__init__(parent)
        self.path: Path | None = None
        self.error: str | None = None
        self._received = 0
        self._total: int | None = None
        self._finished = threading.Event()

        self.title("下載更新")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.attributes("-topmost", True)
        _style(self)

        root = ttk.Frame(self, style="Upd.TFrame", padding=16)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text=f"正在下載 v{release.version}…", style="Upd.TLabel").pack(anchor="w")
        self._bar = ttk.Progressbar(
            root, style="Upd.Horizontal.TProgressbar", length=320, mode="indeterminate"
        )
        self._bar.pack(fill="x", pady=(10, 0))
        self._bar.start(12)
        self._status = ttk.Label(root, text="連線中…", style="UpdDim.TLabel")
        self._status.pack(anchor="w", pady=(6, 0))

        # 關閉視窗不會取消下載（它很快就完成），只是不裝。
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        _place_near(self, parent)
        self.grab_set()

        threading.Thread(
            target=self._work, args=(release,), name="update-download", daemon=True
        ).start()
        self.after(100, self._poll)

    def _work(self, release: update.Release) -> None:
        try:
            self.path = update.stage(release, progress=self._progress)
        except update.UpdateError as exc:
            self.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            self.error = f"下載時發生錯誤：{exc}"
        finally:
            self._finished.set()

    def _progress(self, received: int, total: int | None) -> None:
        self._received = received
        self._total = total

    def _poll(self) -> None:
        if self._finished.is_set():
            self.destroy()
            return
        received, total = self._received, self._total
        if total:
            if str(self._bar.cget("mode")) != "determinate":
                self._bar.stop()
                self._bar.configure(mode="determinate", maximum=total)
            self._bar.configure(value=received)
            self._status.configure(
                text=f"{received / 1_048_576:.1f} / {total / 1_048_576:.1f} MB"
            )
        elif received:
            self._status.configure(text=f"{received / 1_048_576:.1f} MB")
        self.after(100, self._poll)


def ask_update(parent, release: update.Release) -> str:
    dialog = UpdateDialog(parent, release)
    parent.wait_window(dialog)
    return dialog.result


def run_download(parent, release: update.Release) -> tuple[Path | None, str | None]:
    dialog = DownloadDialog(parent, release)
    parent.wait_window(dialog)
    return dialog.path, dialog.error
