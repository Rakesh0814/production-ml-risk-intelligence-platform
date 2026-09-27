@echo off
setlocal

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m training.train_all

if errorlevel 1 (
  echo.
  echo Training failed. Review the error above.
  pause
  exit /b 1
)

echo.
echo Training complete.
echo Start the application with:
echo   .\run_windows.bat
pause
