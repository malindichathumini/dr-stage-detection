@echo off
title DR Stage Detection
cd /d "%~dp0"
if not exist "C:\DR_App\venv\Scripts\activate.bat" (
  echo Python environment not found at C:\DR_App\venv
  echo Create it first: see README.md, section 2.
  pause
  exit /b
)
call "C:\DR_App\venv\Scripts\activate.bat"
echo.
echo Starting the DR Stage Detection app...
echo Your browser opens automatically when it is ready (about 1-2 minutes: 3 models are loaded).
echo Keep this window open while you use the app. Close it to stop the app.
echo.
python app.py
pause
