@echo off
setlocal
cd /d "%~dp0"
echo Requires 64-bit Python 3.12 and an Internet connection.
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv ".venv"
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation complete. Run start.bat to open the program.
pause
exit /b 0
:failed
echo Installation failed. Read the error above. See README_Chinese.md.
pause
exit /b 1
