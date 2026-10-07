"""打包成執行檔時的進入點。

不能直接把 ``src/mapleexp/__main__.py`` 餵給 PyInstaller：那樣它會被當成一支獨立
腳本執行，裡面的相對匯入（``from . import config``）就沒有所屬套件可以解析，
開頭第一行就炸。更糟的是 PyInstaller 的靜態分析也會在同一個地方停下來，於是整個
``mapleexp`` 套件都不會被收進去 —— 症狀是執行檔跑起來少了 tkinter 之類的東西。

所以用這支薄薄的轉接：以套件的方式匯入，分析器才走得進去。
"""

from __future__ import annotations

import sys

from mapleexp.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
