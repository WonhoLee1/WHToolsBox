@echo off
chcp 65001 > nul
echo =======================================================
echo   WHToolsBox Drop Simulator v6 - PyInstaller Build
echo =======================================================
echo.

:: 가상환경 활성화 (현재 설정된 Conda 환경 활성화)
echo [1/3] Activating conda environment 'vdmc'...
call C:\Users\GOODMAN\miniconda3\Scripts\activate.bat vdmc

:: 기존 빌드 폴더 정리
echo [2/3] Cleaning up old build files...
if exist build rmdir /s /q build
if exist dist\WHTools_DropSimulator_v6 rmdir /s /q dist\WHTools_DropSimulator_v6

:: Pyinstaller 실행
echo [3/3] Running PyInstaller with drop_simulator_v6.spec...
pyinstaller --clean drop_simulator_v6.spec

echo.
echo =======================================================
echo   Build Completed!
echo   실행 파일 위치: dist\WHTools_DropSimulator_v6\WHTools_DropSimulator_v6.exe
echo =======================================================
pause
