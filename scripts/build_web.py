"""把 ``src/mapleexp`` 打包成瀏覽器版要載入的 ``web/mapleexp.zip``。

Pyodide 用 ``unpackArchive`` 把 zip 解到它的虛擬檔案系統，之後 ``import mapleexp``
就跟桌面版是同一份程式碼。只有兩處不同：

* 排掉 ``win32/`` 與 ``ui/``（ctypes 綁 Win32、tkinter，瀏覽器裡都沒有），
  以及 ``__pycache__``。
* 用 ``web/shim/`` 裡的替身蓋掉 ``win32/capture.py``，理由寫在那支檔案裡。

``vision/ocr.py``、``identity.py``、``core/session.py`` 這些會匯入 winrt 的模組照樣
放進去 —— 沒人匯入就不會炸，之後要搬 OCR 時再處理。
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mapleexp"
SHIM = ROOT / "web" / "shim" / "mapleexp"
OUT = ROOT / "web" / "mapleexp.zip"

EXCLUDED_DIRS = {"win32", "ui", "__pycache__"}


def main() -> int:
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(SRC)
            if EXCLUDED_DIRS & set(rel.parts[:-1]) or path.suffix == ".pyc":
                continue
            zf.write(path, f"mapleexp/{rel.as_posix()}")
            count += 1
        for path in sorted(SHIM.rglob("*.py")):
            rel = path.relative_to(SHIM)
            zf.write(path, f"mapleexp/{rel.as_posix()}")
            count += 1
    print(f"{OUT.relative_to(ROOT)}: {count} files, {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
