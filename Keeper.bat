@echo off
rem Keeper launcher. Double-click to start without a console window.
rem
rem If the app does not start and shows no error, run:
rem     Keeper.bat console
rem It then runs with a visible console and the error message stays on screen.
rem (Errors are also written to %LOCALAPPDATA%\Keeper\logs\keeper.log.)

chcp 65001 >nul

rem run from our own folder, so it works from shortcuts and other drives too
cd /d "%~dp0"

if not exist "main.py" (
    echo.
    echo ERROR: main.py was not found here:
    echo   %CD%
    echo.
    echo This launcher belongs in the Keeper folder; do not move it.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo ERROR: there is no virtual environment in the project folder.
    echo.
    echo Create it and install the dependencies:
    echo.
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    echo Do not install anything into the system Python.
    echo.
    pause
    exit /b 1
)

rem console mode: python.exe instead of pythonw.exe, so the output is readable
if /i "%~1"=="console" (
    echo Keeper is starting in console mode. This window stays open until it exits.
    echo.
    ".venv\Scripts\python.exe" "main.py"
    echo.
    echo Keeper exited with code %errorlevel%
    pause
    exit /b %errorlevel%
)

rem normal start: pythonw.exe opens no console and start returns at once;
rem a folder or photo given as argument (Open with) is passed on
start "Keeper" ".venv\Scripts\pythonw.exe" "main.py" %*
