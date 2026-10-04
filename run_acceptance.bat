@echo off
setlocal
cd /d "%~dp0"

if not exist "experiments\logs\pytest_tmp" mkdir "experiments\logs\pytest_tmp"
set "TEMP=%~dp0experiments\logs\pytest_tmp"
set "TMP=%~dp0experiments\logs\pytest_tmp"

echo ========================================
echo Hybrid Driving ML Lab - Acceptance Run
echo ========================================

set "PY="
if exist "E:\Anaconda3\python.exe" set "PY=E:\Anaconda3\python.exe"
if not defined PY if exist "C:\Users\Karen Lee\miniconda3\python.exe" set "PY=C:\Users\Karen Lee\miniconda3\python.exe"
if not defined PY set "PY=python"

echo [1/7] Using configured Python:
echo %PY%
%PY% -c "import numpy, pytest; print('NumPy', numpy.__version__); print('pytest', pytest.__version__)"
if errorlevel 1 goto :fail

echo [2/7] Skipping dependency installation (local acceptance mode)...

echo [3/7] Running tests...
%PY% -m pytest -q
if errorlevel 1 goto :fail

echo [4/7] Checking Python compilation...
%PY% -m compileall -q src run.py legacy\pygame_rule
if errorlevel 1 goto :fail

if not exist "experiments\data" mkdir "experiments\data"
if not exist "experiments\models" mkdir "experiments\models"
if not exist "experiments\results" mkdir "experiments\results"

echo [5/7] Collecting demonstration data...
%PY% run.py collect --episodes 20 --seed 7 --scenario mixed --output experiments\data\training_data.npz
if errorlevel 1 goto :fail

echo [6/7] Training neural policy...
%PY% run.py train --data experiments\data\training_data.npz --model experiments\models\neural_model.pkl --epochs 30 --seed 7
if errorlevel 1 goto :fail

echo [7/7] Running rule/neural/hybrid baseline...
%PY% run.py baseline --model experiments\models\neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments\results\baseline.json
if errorlevel 1 goto :fail

echo.
echo SUCCESS: acceptance run completed.
echo Result: experiments\results\baseline.json
echo Model:  experiments\models\neural_model.pkl
echo Data:   experiments\data\training_data.npz
pause
exit /b 0

:fail
echo.
echo FAILED: acceptance run stopped. Check the error above.
pause
exit /b 1
