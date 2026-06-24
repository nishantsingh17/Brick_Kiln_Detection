@echo off
TITLE Brick Kiln Detection System

echo ========================================
echo    Brick Kiln Detection using GEE + YOLO
echo ========================================
echo.

REM Move to script directory
cd /d "%~dp0"

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not added to PATH.
    pause
    exit /b
)

echo Installing dependencies...

if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found.
)

echo.
echo Creating directories...

if not exist outputs mkdir outputs
if not exist models mkdir models

echo.
echo Starting Streamlit app...

if exist app.py (
    python -m streamlit run app.py
) else (
    echo ERROR: app.py not found in:
    echo %cd%
    pause
    exit /b
)

pause