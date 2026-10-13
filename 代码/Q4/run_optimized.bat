@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
chcp 65001 >nul
cd /d "%~dp0"
title Q4 optimized - 25-point concentric rings

where python.exe >nul 2>nul
if errorlevel 1 goto use_py_launcher

python -m optimized.runner
goto finished

:use_py_launcher
where py.exe >nul 2>nul
if errorlevel 1 goto no_python
py -3 -m optimized.runner
goto finished

:no_python
echo Python 3 was not found. Install Python or add it to PATH.
set "RUN_EXIT=9009"
goto hold_window

:finished
set "RUN_EXIT=%errorlevel%"

:hold_window
echo.
if not "%RUN_EXIT%"=="0" echo Run failed. Exit code: %RUN_EXIT%
echo Press any key to close this window.
pause >nul
exit /b %RUN_EXIT%
