@echo off
setlocal
cd /d "%~dp0"

set "CONDA_BAT="
if exist "C:\Users\Karen Lee\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\Users\Karen Lee\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "E:\Anaconda3\condabin\conda.bat" set "CONDA_BAT=E:\Anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
    echo Conda was not found in the known installation paths.
    echo Install Miniconda or Anaconda, then run this file again.
    pause
    exit /b 1
)

echo Using Conda: %CONDA_BAT%
call "%CONDA_BAT%" env update --name hybrid-driving-ml --file environment.yml --prune
if errorlevel 1 (
    echo Failed to create or update the Conda environment.
    pause
    exit /b 1
)

echo.
echo Conda environment ready: hybrid-driving-ml
echo In VS Code, select the interpreter named hybrid-driving-ml.
pause
exit /b 0
