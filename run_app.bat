@echo off
setlocal enabledelayedexpansion
title FetalBrain AI - Deep Learning Diagnostic Portal
echo ======================================================================
echo           🧠 FetalBrain AI - Fetal Neurosonogram System
echo ======================================================================
echo.

cd /d "%~dp0"

REM 1. Detect Python Installation (python or py launcher)
set "PY_CMD="
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python is not found on your computer.
    echo.
    echo Please install Python 3.9, 3.10, or 3.11 from https://www.python.org/downloads/
    echo **CRITICAL**: During Python installation, make sure to check 'Add Python to PATH'!
    echo.
    pause
    exit /b 1
)

echo [✓] Python detected:
%PY_CMD% --version
echo.

REM 2. Create Virtual Environment if not exists
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] First time setup detected! Creating virtual environment 'venv'...
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
    echo.
    echo [INFO] Installing required dependencies (PyTorch, Timm, Flask, ReportLab)...
    echo [INFO] This may take 1-3 minutes depending on your internet connection...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation encountered an issue.
        pause
        exit /b 1
    )
    echo [✓] All dependencies installed successfully!
    echo.
) else (
    call venv\Scripts\activate.bat
)

REM 3. Launch Flask Web Application & Auto-open Browser
echo ======================================================================
echo [INFO] Starting FetalBrain AI Web Server...
echo [INFO] Portal URL: http://127.0.0.1:5000
echo.
echo [INFO] Admin Credentials:
echo        - Login Tab:  Admin Portal
echo        - Username:   admin
echo        - Password:   admin123
echo ======================================================================
echo.

REM Auto-open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

python webapp/app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The application stopped unexpectedly.
    pause
)
