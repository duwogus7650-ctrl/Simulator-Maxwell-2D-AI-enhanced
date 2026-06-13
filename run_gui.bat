@echo off
chcp 65001 >nul
rem motoropt GUI - venv Python으로 실행 (시스템 Python에는 shapely/triangle 없음)
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [오류] 이 폴더에 venv 가상환경이 없습니다.
  echo venv는 용량이 커서 GitHub에 포함되지 않습니다.
  echo venv가 있는 기존 작업폴더에서 실행하거나, 아래로 새로 만드세요:
    echo   py -3.12 -m venv venv
    echo   venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
venv\Scripts\python.exe gui\app.py %*
if errorlevel 1 pause
