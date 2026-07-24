@echo off
setlocal enabledelayedexpansion
title AI Assistant Platform
cd /d "%~dp0"

echo ==============================================
echo   AI Assistant Platform - Startup (Windows)
echo ==============================================
echo.

REM --- 1. Verify Python installation ---------------------------------------
echo [1/7] Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found on PATH.
    echo Please install Python 3.12+ from https://www.python.org/downloads/
    echo and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python --version
echo Python found.
echo.

REM --- 2. Create virtual environment if needed ------------------------------
echo [2/7] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)
echo.

REM --- 3. Activate the environment ------------------------------------------
echo [3/7] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo.

REM --- 4. Install missing dependencies ---------------------------------------
echo [4/7] Installing/verifying dependencies (this may take a minute on first run)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. See the output above.
    pause
    exit /b 1
)
echo Dependencies OK.
echo.

REM --- 5. Verify the .env file -------------------------------------------
echo [5/7] Checking .env file...
if not exist ".env" (
    echo No .env file found - creating one from .env.example
    copy ".env.example" ".env" >nul
    echo.
    echo IMPORTANT: Open the new .env file and add at least one AI provider API key
    echo            (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY) before chatting.
    echo.
) else (
    echo .env file found.
)
echo.

REM --- 6. Verify required API keys -------------------------------------------
echo [6/7] Checking for at least one configured AI provider key...
python -c "from config import settings; import sys; sys.exit(0 if any(settings.providers_available.values()) else 1)" 2>nul
if errorlevel 1 (
    echo WARNING: No AI provider API key is set in .env.
    echo The app will still start, but chat features will be disabled until you add one.
) else (
    echo At least one AI provider is configured.
)
echo.

REM --- 7. Launch the application -----------------------------------------
echo [7/7] Launching AI Assistant Platform...
echo Once started, open your browser to: http://localhost:8000
echo Press CTRL+C in this window to stop the server.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000

if errorlevel 1 (
    echo.
    echo ==============================================
    echo   The application exited with an error.
    echo   See the output above for details.
    echo ==============================================
    pause
    exit /b 1
)

pause
