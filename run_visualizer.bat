@echo off
cd /d "%~dp0\Proje_Kodlari"
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
echo ==============================================
echo  V3 Deep Edge Model - Video Visualizer
echo ==============================================
python visualize_v3.py
pause
