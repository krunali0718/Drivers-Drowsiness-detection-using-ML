@echo off
echo ==============================================================
echo 🚗 ADVANCED DRIVER DROWSINESS SYSTEM LAUNCHER 🚗
echo ==============================================================
echo.

:: Note: We don't need to pip install here anymore since you already installed everything!

echo ==============================================================
echo 🚀 STARTING THE DASHBOARD AND AI CAMERA SEPARATELY 🚀
echo ==============================================================
echo.

:: Launch Streamlit dashboard in the background
start /min cmd /c "streamlit run dashboard.py"

:: Launch the Computer Vision Camera natively
python advanced_drowsiness_test.py
