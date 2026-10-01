@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
"%PY%" run.py %*
exit /b %errorlevel%
