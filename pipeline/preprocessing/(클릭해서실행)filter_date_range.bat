@echo off
setlocal
cd /d "%~dp0"

python "%~dp0filter_date_range.py"

echo.
pause
