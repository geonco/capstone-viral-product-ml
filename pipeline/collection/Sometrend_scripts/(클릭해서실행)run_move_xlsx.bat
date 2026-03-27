@echo off
setlocal
cd /d "%~dp0"

python "%~dp0move_xlsx_to_sometrend2.py" %*

echo.
pause
