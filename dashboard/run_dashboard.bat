@echo off
REM Windows launcher -- Ghana Malaria Dashboard
cd /d "%~dp0"
echo Starting Ghana Malaria Dashboard...
echo Open http://127.0.0.1:8050 in your browser
start "" "http://127.0.0.1:8050"
timeout /t 1 /nobreak >nul
python app.py
pause
