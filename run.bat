@echo off
title AI Disease Prediction System
echo ==============================================
echo    Starting AI Disease Prediction System...
echo ==============================================
echo.

set PYTHON_PATH="C:\Users\Aakash Jana\AppData\Local\Programs\Python\Python315\python.exe"

:: Open the browser URL in parallel after a 2-second delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8501"

if exist %PYTHON_PATH% (
    %PYTHON_PATH% -m streamlit run app.py --server.headless=false
) else (
    python -m streamlit run app.py --server.headless=false
)

pause
