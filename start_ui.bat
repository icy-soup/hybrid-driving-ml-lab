@echo off
setlocal
cd /d "%~dp0"

set "PY="
if exist "E:\Anaconda3\python.exe" set "PY=E:\Anaconda3\python.exe"
if not defined PY if exist "C:\Users\Karen Lee\miniconda3\python.exe" set "PY=C:\Users\Karen Lee\miniconda3\python.exe"
if not defined PY set "PY=python"

echo Starting Hybrid Driving ML Lab UI (MPC audit mode)...
echo Close the window or press ESC to exit.
%PY% run.py ui --policy mpc --scenario mixed --seed 7 --fps 30
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo UI exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
