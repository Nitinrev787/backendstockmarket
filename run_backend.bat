@echo off
echo ===============================
echo Starting FastAPI Stock Backend
echo ===============================

:: Move to backend folder
cd /d %~dp0

:: Optional: Activate venv (uncomment if you have one)
:: call venv\Scripts\activate

:: Install dependencies (first run only)
pip install fastapi uvicorn pandas numpy yfinance scikit-learn

:: Run FastAPI server
uvicorn main:app --reload

pause
