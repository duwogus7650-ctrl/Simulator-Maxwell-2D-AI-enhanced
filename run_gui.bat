@echo off
rem motoropt GUI — venv Python으로 실행 (시스템 Python에는 shapely/triangle 없음)
cd /d "%~dp0"
venv\Scripts\python.exe gui\app.py %*
if errorlevel 1 pause
