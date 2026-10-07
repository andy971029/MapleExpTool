@echo off
rem Launcher: no install needed, just puts src on PYTHONPATH.
rem   run.cmd calibrate
rem   run.cmd track --label <map-name>
rem   run.cmd doctor
rem Keep this file ASCII-only: cmd.exe reads .cmd with the ANSI codepage,
rem so non-ASCII comments get mangled and printed as errors.
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
set "PYTHONUTF8=1"
if exist "%ROOT%.venv\Scripts\python.exe" (
    "%ROOT%.venv\Scripts\python.exe" -m mapleexp %*
) else (
    python -m mapleexp %*
)
endlocal
