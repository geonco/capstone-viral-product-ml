@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Enter the XLSX file path.
    set /p INPUT_PATH=Path: 
    if "%INPUT_PATH%"=="" (
        echo Input was empty.
        echo.
        pause
        exit /b 1
    )

    set "INPUT_PATH=%INPUT_PATH:"=%"
    python "%~dp0xlsx_to_csv.py" "%INPUT_PATH%"
) else (
    python "%~dp0xlsx_to_csv.py" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo Conversion failed. Exit code: %EXIT_CODE%
) else (
    echo Conversion completed.
)

pause
exit /b %EXIT_CODE%
