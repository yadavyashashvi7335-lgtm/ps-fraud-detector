@echo off
chcp 65001 >nul
title PS Fraud Detector - Backend

echo ========================================
echo  PS Fraud Detector Backend Starting...
echo ========================================
echo.

cd /d "%~dp0"

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python from python.org
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist ".venv" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
)

:: Activate venv
echo [2/4] Activating environment...
call .venv\Scripts\activate.bat

:: Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt

:: Train baseline model
echo [4/4] Training baseline model (first time only)...
python scripts/train_baseline.py

echo.
echo ========================================
echo  Server starting at: http://localhost:8000
echo  Frontend:           http://localhost:8000/ui
echo ========================================
echo  Press Ctrl+C to stop the server.
echo ========================================
echo.

uvicorn app.main:app --reload --port 8000

pause
