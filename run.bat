@echo off
rem Launch the avatar app inside the .venv created by setup.bat.
rem All arguments are forwarded to main.py, e.g.:
rem   run.bat --mock
rem   run.bat --char my_char.png --driver hybrid --output window,virtualcam
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
