@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo 병합할 키워드 폴더 경로를 입력하세요.
    echo 예: Sometrend\칸쵸
    set /p INPUT_PATH=Path:
    if "%INPUT_PATH%"=="" (
        echo 입력이 없습니다.
        echo.
        pause
        exit /b 1
    )

    set "INPUT_PATH=%INPUT_PATH:"=%"
    python "%~dp0merge_sometrend.py" "%INPUT_PATH%"
) else (
    python "%~dp0merge_sometrend.py" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo 병합 실패. Exit code: %EXIT_CODE%
) else (
    echo 병합 완료.
)

pause
exit /b %EXIT_CODE%
