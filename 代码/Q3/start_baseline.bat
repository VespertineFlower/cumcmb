@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python -m baseline.runner --mode q3 %*
) else (
    py -3 -m baseline.runner --mode q3 %*
)
set "run_exit=%errorlevel%"
echo.
echo Baseline 程序已结束，退出码：%run_exit%
pause >nul
exit /b %run_exit%
