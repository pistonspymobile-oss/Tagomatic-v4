# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for TagOmatic v4.2 (macOS)
# Build: pyinstaller TagOmatic-v4.2-mac.spec
# Note: ExifTool is expected to be installed via Homebrew (brew install exiftool)

block_cipher = None

a = Analysis(
    ['car_identifier_gui_qt_v3_final_merged.py', 'race_metadata_parser.py', 'secret_cloud.py'],
    pathex=[],
    binaries=[
        # Note: ExifTool is not bundled - it's expected to be in system PATH
        # (installed via Homebrew: brew install exiftool)
        # This keeps the executable smaller and uses system ExifTool
    ],
    datas=[
        ('assets', 'assets'),  # Icons, logos, and UI images
        # ONNX model files for intelligent vehicle detection (optional - only include if file exists)
        # Note: Users can place .onnx files next to the executable if needed
        # ('yolov10n.onnx', '.'),  # Uncomment if available
        # Developer-approved knowledge base (optional - can also be placed next to exe)
        # ('approved_knowledge_base.json', '.'),  # Uncomment to bundle approved KB with exe
    ],
    hiddenimports=[
        # Core GUI framework
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageFile',
        # LLM client
        'ollama',
        # Optional: ONNX Runtime (for intelligent cropping)
        'onnxruntime',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'numpy',
        # HTTP requests for model unloading
        'requests',
        # PDF parsing for race metadata
        'pdfplumber',
        # Standard library (usually auto-detected, but explicit for safety)
        'base64',
        'json',
        'io',
        'threading',
        'pathlib',
        're',
        'subprocess',
        'tempfile',
        'time',
        'os',
        'sys',
        'datetime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Large unnecessary packages
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        # Not used in app
        'flask',
        'werkzeug',
        'watchdog',  # Not the library, just Qt timer variable name
        'torch',
        'torchvision',
        'ultralytics',  # Only needed for .pt to .onnx conversion (optional)
        'scikit-learn',
        'sklearn',
        'ddgs',
        'beautifulsoup4',
        'bs4',
        'lxml',
        'playwright',
        # Cloud/API packages (not used in local mode)
        # Note: requests is now needed for model unloading functionality
        'openai',
        'google',
        'anthropic',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TagOmatic-v4.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX compression disabled on macOS (Apple restrictions)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,  # Set to your signing identity if code signing
    entitlements_file=None,  # Path to entitlements.plist if needed
    icon=None,  # Icon disabled for now (can add .icns file later if needed)
    version=None,  # Can add version info if needed
)

