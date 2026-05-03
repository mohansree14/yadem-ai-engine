@echo off
echo Starting YADEM AI Engine Demo Environment...
echo ------------------------------------------

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11+.
    pause
    exit /b
)

:: Start Backend in a new window
echo [1/2] Starting API Backend (Port 8000)...
start "YADEM Backend" cmd /c "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

:: Start Frontend Server
echo [2/2] Starting Frontend (Port 8080)...
echo Launching browser to http://localhost:8080/index.html
start http://localhost:8080/index.html
cd frontend && python -m http.server 8080

pause
