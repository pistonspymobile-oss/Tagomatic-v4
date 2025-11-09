# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for TagOmatic v4.2
# Build: pyinstaller TagOmatic-v4.2.spec

block_cipher = None

a = Analysis(
    ['car_identifier_gui_qt_v3_final_merged.py', 'race_metadata_parser.py'],
    pathex=[],
    binaries=[
        # ExifTool - try exe first, fallback to perl wrapper
        ('exiftool.exe', '.'),
        # ExifTool Perl wrapper and dependencies (if exe doesn't work)
        ('exiftool_files/exiftool.pl', 'exiftool_files'),
        ('exiftool_files/perl.exe', 'exiftool_files'),
        ('exiftool_files/perl532.dll', 'exiftool_files'),
        ('exiftool_files/libgcc_s_seh-1.dll', 'exiftool_files'),
        ('exiftool_files/liblzma-5__.dll', 'exiftool_files'),
        ('exiftool_files/libstdc++-6.dll', 'exiftool_files'),
        ('exiftool_files/libwinpthread-1.dll', 'exiftool_files'),
    ],
    datas=[
        ('assets', 'assets'),
        ('exiftool_files/lib', 'exiftool_files/lib'),
        ('exiftool_files/LICENSE', 'exiftool_files'),
        ('exiftool_files/readme_windows.txt', 'exiftool_files'),
        ('exiftool_files/windows_exiftool.txt', 'exiftool_files'),
        # ONNX model files for intelligent vehicle detection (optional but recommended)
        ('yolov10n.onnx', '.'),  # Preferred ONNX model
        # Note: yolov8n.onnx can be added if available, or .pt files if conversion is needed
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
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/tagomatic_main.ico',  # Use main icon
    version=None,  # Can add .rc file for version info if needed
)

