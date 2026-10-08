@echo off
rem Build web\mapleexp.zip and serve the web prototype on http://localhost:8765
rem getDisplayMedia needs a secure context; http://localhost counts as one.
rem Keep this file ASCII-only: cmd.exe reads .cmd with the ANSI codepage,
rem so non-ASCII comments get mangled and printed as errors.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
if not exist "%PY%" set "PY=python"

"%PY%" "%ROOT%\scripts\build_web.py" || exit /b 1
echo.
echo Open http://localhost:8765/ in Chrome or Edge. Ctrl+C to stop.
"%PY%" -m http.server 8765 --bind 127.0.0.1 --directory "%~dp0"
endlocal
