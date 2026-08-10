@echo off
rem Launch the avatar app inside the .venv created by setup.bat.
rem
rem With arguments, they are forwarded to main.py unchanged, e.g.:
rem   run.bat --mock
rem   run.bat --char characters\my_char.png --driver hybrid --output window,virtualcam
rem With no arguments (e.g. double-click), an interactive menu asks for the
rem mode and lets you pick a character from the characters\ folder.
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)

if not "%~1"=="" (
    ".venv\Scripts\python.exe" main.py %*
    if errorlevel 1 pause
    exit /b
)

echo Select a mode:
echo   1. webcam      - your webcam drives the avatar
echo   2. procedural  - idle animation + control server, no webcam needed
echo   3. hybrid      - webcam tracking, procedural channels can override
echo   4. mock        - pipeline test, no GPU/models/character needed
set "MODE="
set /p "MODE=Mode [1-4]: "

if "!MODE!"=="4" (
    ".venv\Scripts\python.exe" main.py --mock
    if errorlevel 1 pause
    exit /b
)
set "DRIVER="
if "!MODE!"=="1" set "DRIVER=webcam"
if "!MODE!"=="2" set "DRIVER=procedural"
if "!MODE!"=="3" set "DRIVER=hybrid"
if not defined DRIVER (
    echo Invalid mode "!MODE!".
    pause
    exit /b 1
)

echo.
echo Select a character:
set COUNT=0
for %%F in ("characters\*.png") do (
    set /a COUNT+=1
    set "CHAR_!COUNT!=%%~fF"
    echo   !COUNT!. %%~nxF
)
if %COUNT%==0 (
    echo   No character images found.
    echo   Put a 512x512 RGBA PNG in the characters\ folder and run again.
    pause
    exit /b 1
)
set "SEL="
set /p "SEL=Character [1-%COUNT%]: "
set "CHARFILE="
if defined SEL call set "CHARFILE=%%CHAR_!SEL!%%"
if not defined CHARFILE (
    echo Invalid character "!SEL!".
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py --char "!CHARFILE!" --driver !DRIVER!
if errorlevel 1 pause
