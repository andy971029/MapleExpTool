@echo off
rem Publish a new version to GitHub Releases: build + sha256 + tag + upload.
rem Needs the GitHub CLI (winget install GitHub.cli; gh auth login).
rem Keep this file ASCII-only: cmd.exe reads .cmd with the ANSI codepage,
rem so non-ASCII comments get mangled and printed as errors.
rem
rem   release.cmd                       release notes generated from commits
rem   release.cmd --notes "what changed"
rem   release.cmd --dry-run             build and check only, publish nothing
rem   release.cmd --skip-build          reuse dist\MapleExpTool.exe
rem
rem The version comes from src\mapleexp\__init__.py (__version__); bump it first.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
if not exist "%PY%" set "PY=python"
"%PY%" "%ROOT%scripts\release.py" %*
endlocal
