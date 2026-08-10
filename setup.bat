@echo off
rem ============================================================
rem  One-shot Windows setup for the THA3 avatar wrapper.
rem
rem  Creates .venv, installs Python dependencies, installs
rem  PyTorch (CUDA build if an NVIDIA GPU is found), clones
rem  talking-head-anime-3-demo next to this repo, downloads the
rem  THA3 model weights (~800 MB), and verifies everything.
rem
rem  Usage:  setup.bat [extra args passed to scripts\setup_tha3.py]
rem          e.g.  setup.bat --torch cpu
rem                setup.bat --skip-models
rem ============================================================
setlocal
cd /d "%~dp0"

rem --- find Python ---------------------------------------------------
set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [setup] ERROR: Python not found on PATH.
        echo [setup] Install Python 3.10-3.12 from https://www.python.org/downloads/
        echo [setup] and check "Add python.exe to PATH" in the installer.
        goto :fail
    )
    set "PYTHON=py -3"
)

%PYTHON% -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] else 1)"
if errorlevel 1 (
    echo [setup] ERROR: Python 3.9 or newer is required.
    goto :fail
)
%PYTHON% -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo [setup] WARNING: Python newer than 3.12 detected - mediapipe may
    echo [setup] not have wheels for it yet. If webcam tracking fails to
    echo [setup] install, use Python 3.12.
)

rem --- create virtual environment ------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment in .venv ...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [setup] using existing .venv
)
set "VPY=.venv\Scripts\python.exe"

rem --- install dependencies ------------------------------------------
echo [setup] upgrading pip ...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [setup] installing requirements.txt ...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem --- PyTorch + THA3 repo + model weights ---------------------------
"%VPY%" scripts\setup_tha3.py %*
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Setup finished. Quick start (or use run.bat):
echo.
echo    run.bat --mock
echo        pipeline test without the GPU models
echo.
echo    run.bat --char ..\talking-head-anime-3-demo\data\images\lambda_00.png --driver webcam
echo        webcam-driven VTuber (add --output window,virtualcam for OBS)
echo.
echo    run.bat --char ..\talking-head-anime-3-demo\data\images\lambda_00.png --driver procedural
echo        AI-controllable avatar (control server on 127.0.0.1:9535)
echo.
echo  For the OBS virtual camera, install OBS Studio 26+ once:
echo    https://obsproject.com/
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [setup] Setup FAILED - see the messages above.
pause
exit /b 1
