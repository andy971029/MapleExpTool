# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 設定：打包成免安裝的單一執行檔。

用法（在專案根目錄）::

    .venv\\Scripts\\pyinstaller.exe MapleExpTool.spec --noconfirm

產物是 ``dist/MapleExpTool.exe``，複製到任何一台 Windows 上雙擊就能跑，不需要
裝 Python、也不需要裝任何套件。

幾個不明顯但必要的設定：

* ``collect_all('winrt')`` —— OCR 走的是 Windows 內建引擎，透過 winrt 綁定呼叫。
  那些綁定是一堆動態載入的子模組（``winrt.windows.media.ocr`` 等），PyInstaller
  的靜態分析掃不到，不整包收進來的話執行檔會在辨識中文時直接失效。
* ``collect_all('windows_capture')`` —— 擷取後端是 Rust 寫的 ``.pyd``，同理。
* ``builtin_templates.json`` —— 內建的數字字元模板。少了它，使用者第一次開就得
  自己標字元，整個「開箱即用」就不成立了。
* 進入點是 ``packaging/entry.py`` 而不是 ``__main__.py`` —— 理由寫在那支檔案裡。
* ``hook_cv2_stub.py`` —— 把 OpenCV 排掉並放一個替身，省下 38MB。理由同上，
  寫在那支檔案裡。
* ``console=False`` —— 雙擊開浮窗時不要有黑色的 cmd 視窗。``doctor``／``report``
  這些要印字的子命令由 ``__main__._setup_console`` 處理：從終端機執行就用
  AttachConsole 接回那個終端機，沒有終端機就自己開一個並在結束前停住。
  開始追蹤前的錯誤（找不到視窗之類）改用對話框，不會無聲消失。
"""

from PyInstaller.utils.hooks import collect_all

datas = [
    ("src/mapleexp/builtin_templates.json", "mapleexp"),
    # 浮窗與存檔對話框會拿它當視窗圖示
    ("assets/icon.ico", "assets"),
]
binaries = []
hiddenimports = []

for package in ("winrt", "windows_capture", "lz4"):
    extra_datas, extra_binaries, extra_hidden = collect_all(package)
    datas += extra_datas
    binaries += extra_binaries
    hiddenimports += extra_hidden

analysis = Analysis(
    ["packaging/entry.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["packaging/hook_cv2_stub.py"],
    # 這支程式不用這些，排掉可以少幾十 MB。opencv／Pillow／UnityPy 是開發時
    # 試東西留在 venv 裡的，專案本身一行都沒用到。
    excludes=[
        "pytest", "unittest", "pydoc_data", "setuptools", "pip", "PyInstaller",
        "cv2", "PIL", "UnityPy", "scipy", "pandas", "matplotlib",
        "pyfmodex", "texture2ddecoder", "etcpak", "astc_encoder",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="MapleExpTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon="assets/icon.ico",
)
