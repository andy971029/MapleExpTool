@echo off
rem Build the portable executable: dist\MapleExpTool.exe
rem Keep this file ASCII-only: cmd.exe reads .cmd with the ANSI codepage,
rem so non-ASCII comments get mangled and printed as errors.
rem
rem   build.cmd           rebuild icon + exe
rem   build.cmd icon      rebuild the icon only
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
if not exist "%PY%" set "PY=python"

echo [1/2] building icon from assets\Logo.jpeg
"%PY%" "%ROOT%scripts\make_icon.py" || exit /b 1
if /i "%~1"=="icon" goto :done

echo [2/2] building dist\MapleExpTool.exe
"%PY%" -m PyInstaller "%ROOT%MapleExpTool.spec" --noconfirm ^
    --distpath "%ROOT%dist" --workpath "%ROOT%build" || exit /b 1

:done
echo.
echo Done. Portable executable: %ROOT%dist\MapleExpTool.exe
endlocal
