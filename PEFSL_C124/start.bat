@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" launch.py %*
if errorlevel 1 pause
exit /b
:missing
echo Please install 64-bit Python 3.12 and run install.bat first.
pause
exit /b 1
