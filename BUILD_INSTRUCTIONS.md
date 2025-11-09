# Building TagOmatic v4.2 Executable

## Prerequisites

1. **Python 3.12+** (tested with Python 3.12)
2. **PyInstaller**: `pip install pyinstaller`
3. **Required Python packages** (install from requirements.txt):
   ```bash
   pip install PySide6 Pillow ollama onnxruntime numpy
   ```

## Quick Build (Windows)

Run the provided batch script:
```bash
build_exe.bat
```

## Manual Build

1. **Install PyInstaller** (if not already installed):
   ```bash
   pip install pyinstaller
   ```

2. **Build the executable**:
   ```bash
   pyinstaller TagOmatic-v4.2.spec --clean
   ```

3. **Output**: The executable will be in `dist/TagOmatic-v4.2.exe`

## Building Mac Executable (No Mac Required!)

**You can build Mac executables from Windows using GitHub Actions!**

### Option 1: GitHub Actions (Recommended - No Mac Needed)

This is the easiest way to build Mac executables without having a Mac:

1. **Push your code to GitHub** (create a repository if you haven't already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **GitHub Actions will automatically build** when you push code that changes:
   - `car_identifier_gui_qt_v3_final_merged.py`
   - `race_metadata_parser.py`
   - `TagOmatic-v4.2-mac.spec`
   - `requirements.txt`

3. **Download the executable**:
   - Go to your GitHub repository
   - Click on the **Actions** tab
   - Find the latest workflow run (should be "Build Mac Executable")
   - Wait for it to complete (~5-10 minutes)
   - Click on the workflow run
   - Scroll down to **Artifacts**
   - Download `TagOmatic-v4.2-mac`

4. **Manual trigger** (if needed):
   - Go to Actions tab
   - Click "Build Mac Executable" workflow
   - Click "Run workflow" button
   - Select branch and click "Run workflow"

### Option 2: Build on Mac Directly

If you have access to a Mac, you can build locally:

1. **Prerequisites on Mac**:
   - Python 3.12+ installed
   - Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)

2. **Install ExifTool**:
   ```bash
   brew install exiftool
   ```

3. **Run the build script**:
   ```bash
   chmod +x build_exe_mac.sh
   ./build_exe_mac.sh
   ```

   Or build manually:
   ```bash
   pip3 install -r requirements.txt
   pip3 install pyinstaller
   pyinstaller TagOmatic-v4.2-mac.spec --clean
   ```

4. **Output**: The executable will be in `dist/TagOmatic-v4.2`

### Mac Build Notes

- **ExifTool**: Not bundled - must be installed on the system (via Homebrew)
- **Icon**: Uses PNG format (PyInstaller handles conversion)
- **Code Signing**: Optional. Unsigned apps may require right-click → Open on first launch
- **File Size**: Typically 150-250MB (similar to Windows)
- **UPX Compression**: Disabled on macOS (Apple restrictions)

### Mac Troubleshooting

**"exiftool not found" error**:
- Install ExifTool: `brew install exiftool`
- Verify installation: `which exiftool` and `exiftool -ver`

**"App is damaged" or "can't be opened"**:
- Right-click the app → Open (first time only)
- Or code sign the executable (requires Apple Developer account)

**Large executable size**:
- Normal for PyInstaller bundles
- Includes PySide6, ONNX Runtime, and all dependencies

## What's Included

The spec file (`TagOmatic-v4.2.spec`) includes:

- **Main script**: `car_identifier_gui_qt_v3_final_merged.py`
- **Assets folder**: All logos, icons, and images from `assets/`
- **ExifTool**: 
  - `exiftool.exe` (primary)
  - Perl wrapper and dependencies in `exiftool_files/` (fallback)
  - All Perl libraries and DLLs
- **Python dependencies**: PySide6, Pillow, ollama, onnxruntime, numpy, etc.

## Optional Model Files

The following model files are **NOT** included (optional, can be added later):
- `yolov8n.pt` / `yolov10n.pt` - PyTorch YOLO models
- `yolov8n.onnx` / `yolov10n.onnx` - ONNX models (auto-converted if .pt found)
- `logo_classifier.pth` / `logo_classifier_cars.pth` - Logo classifier models

If users want ONNX cropping, they can place these files next to the executable.

## Testing the Executable

1. Navigate to the `dist` folder
2. Run `TagOmatic-v4.2.exe`
3. The app should start with all assets and exiftool working

## Troubleshooting

### "exiftool not found" error
- Ensure `exiftool.exe` is in the same directory as the spec file
- Check that `exiftool_files/` folder exists with all dependencies

### "Asset not found" error
- Ensure `assets/` folder exists with all logo/image files
- Check that `_resource_path()` function can find assets in `_MEIPASS`

### Large executable size
- Normal: PyInstaller bundles all dependencies
- Typical size: 100-200MB due to PySide6, ONNX Runtime, and ExifTool
- Use `--exclude-module` in spec file to reduce size if needed

### Missing DLL errors
- Ensure all DLLs in `exiftool_files/` are included
- Check that ONNX Runtime DLLs are automatically included by PyInstaller

## Build Notes

### Windows Build
- **Console window**: Disabled (`console=False`) for clean GUI-only app
- **UPX compression**: Enabled to reduce file size (may slow startup slightly)
- **Icon**: Uses `assets/tagomatic_main.ico`
- **Single file**: Creates one executable (no separate DLLs needed)

### Mac Build
- **Console window**: Disabled (`console=False`) for clean GUI-only app
- **UPX compression**: Disabled (Apple restrictions on macOS)
- **Icon**: Uses `assets/logo.png` (PNG format)
- **Single file**: Creates one executable (no separate libraries needed)
- **ExifTool**: Uses system installation (not bundled)

