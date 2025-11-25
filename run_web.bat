@echo off
REM AI-Powered Refugee Crisis Intelligence System - Web Interface Launcher

echo ================================================================================
echo    AI-Powered Refugee Crisis Intelligence System - Web Interface
echo ================================================================================
echo.
echo Starting web server...
echo.
echo The dashboard will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

cd web
python app.py
