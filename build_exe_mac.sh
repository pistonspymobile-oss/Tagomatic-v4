#!/bin/bash
# Build script for TagOmatic v4.2 executable (macOS)
# Run this script on a Mac to build the executable locally

echo "========================================"
echo "Building TagOmatic v4.2 Executable (macOS)"
echo "========================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is for macOS only."
    echo "   For building from Windows, use GitHub Actions instead."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.12+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "⚠️  Warning: Python $PYTHON_VERSION detected. Python 3.12+ is recommended."
fi

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install PyInstaller. Please install manually: pip3 install pyinstaller"
        exit 1
    fi
fi

# Check if ExifTool is installed
if ! command -v exiftool &> /dev/null; then
    echo "ExifTool not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    brew install exiftool
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install ExifTool. Please install manually: brew install exiftool"
        exit 1
    fi
else
    echo "✅ ExifTool found: $(which exiftool)"
    exiftool -ver
fi

# Check if required dependencies are installed
echo ""
echo "Checking Python dependencies..."
echo "Installing requirements from requirements.txt..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some dependencies may have failed to install."
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist __pycache__ *.spec.bak

# Build executable
echo ""
echo "Building executable..."
echo "This may take several minutes..."
echo ""
pyinstaller TagOmatic-v4.2-mac.spec --clean --noconfirm

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi

# Verify executable
if [ -f "dist/TagOmatic-v4.2" ]; then
    echo ""
    echo "========================================"
    echo "✅ Build Complete!"
    echo "========================================"
    echo ""
    echo "Executable location: dist/TagOmatic-v4.2"
    echo "File size: $(du -h dist/TagOmatic-v4.2 | cut -f1)"
    echo ""
    echo "To test the executable:"
    echo "  1. Navigate to dist folder"
    echo "  2. Run: ./TagOmatic-v4.2"
    echo ""
    echo "Note: The executable is standalone and includes all dependencies."
    echo "      ExifTool is expected to be installed on the system (via Homebrew)."
    echo ""
    
    # Optional: Code signing reminder
    echo "💡 Optional: To code sign the executable for distribution:"
    echo "   codesign --deep --force --verify --verbose --sign \"Developer ID Application: Your Name\" dist/TagOmatic-v4.2"
    echo ""
else
    echo ""
    echo "❌ Build completed but executable not found!"
    exit 1
fi

