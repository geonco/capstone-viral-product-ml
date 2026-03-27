@echo off
setlocal
cd /d "%~dp0"

python "%~dp0build_mention_features.py"

echo.
pause
