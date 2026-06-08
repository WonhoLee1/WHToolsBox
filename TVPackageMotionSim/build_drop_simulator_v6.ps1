# [WHTOOLS] Drop Simulator v6 - PyInstaller Build Script for PowerShell
# UTF-8 Encoding Enforcement
$OutputEncoding = [System.Text.UTF8Encoding]::new()

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  WHToolsBox Drop Simulator v6 - PyInstaller Build (PS)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Set current location to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir) { 
    Set-Location $ScriptDir 
}

# [1/3] Activate conda environment
Write-Host "[1/3] Activating conda environment 'vdmc'..."
if (Test-Path "C:\Users\GOODMAN\miniconda3\condabin\conda.bat") {
    & "C:\Users\GOODMAN\miniconda3\condabin\conda.bat" activate vdmc
} else {
    conda activate vdmc
}

# [2/3] Clean up old build files
Write-Host "[2/3] Cleaning up old build files..."
if (Test-Path "build") {
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    if (Test-Path "build") {
        Write-Error "Failed to clean 'build' directory. It may be locked by another process."
        Read-Host "Press Enter to exit..."
        exit 1
    }
}
if (Test-Path "dist\WHTools_DropSimulator_v6") {
    Remove-Item -Recurse -Force dist\WHTools_DropSimulator_v6 -ErrorAction SilentlyContinue
    if (Test-Path "dist\WHTools_DropSimulator_v6") {
        Write-Error "Failed to clean 'dist\WHTools_DropSimulator_v6' directory. It may be locked by another process."
        Read-Host "Press Enter to exit..."
        exit 1
    }
}

# [3/3] Run Pyinstaller
Write-Host "[3/3] Running PyInstaller inside 'vdmc' conda environment..."
#conda run -n vdmc pyinstaller --clean drop_simulator_v6.spec
conda run -n vdmc pyinstaller --clean --noconfirm drop_simulator_v6.spec

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Build Completed!" -ForegroundColor Green
Write-Host "  Executable location: dist\WHTools_DropSimulator_v6\WHTools_DropSimulator_v6.exe" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Read-Host "Press Enter to continue..."
