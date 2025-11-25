@echo off
echo ========================================
echo Installing AI-Powered Refugee Crisis Intelligence System
echo ========================================
echo.

REM Stop current installation if running
echo Press Ctrl+C to cancel the current installation first, then run this script
pause

echo Step 1: Installing core dependencies...
pip install --no-cache-dir python-dotenv pydantic typing-extensions

echo.
echo Step 2: Installing Google AI dependencies...
pip install --no-cache-dir google-generativeai google-cloud-storage

echo.
echo Step 3: Installing data processing libraries...
pip install --no-cache-dir pandas numpy

echo.
echo Step 4: Installing visualization libraries...
pip install --no-cache-dir plotly matplotlib

echo.
echo Step 5: Installing other utilities...
pip install --no-cache-dir requests twilio python-dateutil tqdm Pillow

echo.
echo Step 6: Installing machine learning libraries (optional, may take time)...
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir scikit-learn

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Add your GEMINI_API_KEY to .env
echo 3. Run: python main.py --mode demo
echo.
pause
