@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo =======================================================
echo   WHToolsBox Drop Simulator v6 - PyInstaller Build
echo =======================================================
echo.

:: Activate conda environment
echo [1/3] Activating conda environment 'vdmc'...
call C:\Users\GOODMAN\miniconda3\Scripts\activate.bat vdmc

:: Clean up old build files
echo [2/3] Cleaning up old build files...
if exist build (
    rmdir /s /q build
    if exist build (
        echo.
        echo ❌ [ERROR] Failed to clean 'build' directory. It may be locked by another process.
        pause
        exit /b 1
    )
)
if exist dist\WHTools_DropSimulator_v6 (
    rmdir /s /q dist\WHTools_DropSimulator_v6
    if exist dist\WHTools_DropSimulator_v6 (
        echo.
        echo ❌ [ERROR] Failed to clean 'dist\WHTools_DropSimulator_v6' directory.
        echo Please close any running simulator instances, terminals, or explorer windows in this folder.
        pause
        exit /b 1
    )
)

:: Run Pyinstaller
echo [3/3] Running PyInstaller with drop_simulator_v6.spec...
::pyinstaller --clean drop_simulator_v6.spec
pyinstaller --clean --noconfirm drop_simulator_v6.spec

echo.
echo =======================================================
echo   Build Completed!
echo   Executable location: dist\WHTools_DropSimulator_v6\WHTools_DropSimulator_v6.exe
echo =======================================================
pause
