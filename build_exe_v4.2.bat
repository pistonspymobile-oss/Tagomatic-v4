@echo off
REM Build script for TagOmatic v4.2 executable
echo ========================================
echo Building TagOmatic v4.2 Executable
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller. Please install manually: pip install pyinstaller
        pause
        exit /b 1
    )
)

REM Check if required dependencies are installed
echo Checking dependencies...
echo Installing minimal requirements from requirements.txt...
pip install -r requirements.txt

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

REM Build executable
echo.
echo Building executable...
echo This may take several minutes...
echo.
pyinstaller TagOmatic-v4.2.spec --clean

if errorlevel 1 (
    echo.
    echo Build failed! Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: dist\TagOmatic-v4.2.exe
echo.
echo To test the executable:
echo   1. Navigate to dist folder
echo   2. Run TagOmatic-v4.2.exe
echo.
echo Note: The exe is standalone and includes all dependencies.
echo.
pause



