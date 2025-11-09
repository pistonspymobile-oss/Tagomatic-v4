# TagOmatic v4.1 (Clean Rewrite)

Run dev app:

```
python -m app.main
```

Requirements:
- Python 3.11+
- `pip install -r requirements.txt`

Features (phase 1):
- Cloud Backends dialog (OpenAI, Gemini, External Ollama, OpenAI-compatible)
- OpenAI-compatible and External Ollama routing live
- Strict 7-line car identification pipeline
- Results list with thumbnails
- Best-effort EXIF write (JPEG)

Packaging (later phase):
- PyInstaller spec and Windows EXE

Pistonspy : TagOmatic (v4.0)

Overview

TagOmatic is a local, privacy‑first photo tagger focused on cars and motorsport images. It uses Ollama with vision‑capable LLMs to extract Make, Model, Color, Logos, License Plate, and a concise descriptive summary, then writes standards‑compliant IPTC/EXIF metadata via ExifTool.

Key Features

- Local inference (no cloud); Ollama models
- Two‑stage parsing with optional second‑pass verification
- Batch processing with skip/overwrite controls
- Streamlined PySide6 UI with recent results and details viewer

What's in this folder

- car_identifier_gui_qt_v3_final.py – main application source
- dist/TagOmatic-v4.0.exe – packaged Windows build
- TagOmatic-v4.0.spec – PyInstaller build spec
- exiftool_files/ and exiftool.exe – embedded ExifTool runtime
- requirements.txt – Python dependencies

Prerequisites

- Windows 10/11
- Python 3.10+ (for running from source)
- Ollama installed and running (`https://ollama.com/download`)
- Vision models installed (example):
  - `ollama pull qwen2.5-vl:7b` (or similar vision model)
  - `ollama pull gemma3:12b` (if you prefer a larger model)

Quick Start (packaged EXE)

1) Open `dist/` and run `TagOmatic-v4.0.exe`
2) Load an image or choose a folder to batch
3) Optionally enable 2nd pass verification
4) Approve results to write IPTC/EXIF

Run from Source

1) Install Python dependencies:

   pip install -r requirements.txt

2) Ensure ExifTool is available. This folder ships a Windows ExifTool bundle under `exiftool_files/` with a launcher `exiftool.exe` at the repo root.

3) Start Ollama (GPU optional). If you have NVIDIA GPU and CUDA set up:

   powershell -ExecutionPolicy Bypass -File .\start_ollama_gpu.ps1

   Otherwise simply start the Ollama service from your OS or terminal.

4) Pull the models (one‑time):

   ollama pull qwen2.5-vl:7b
   ollama pull gemma3:12b

5) Run the app:

   python car_identifier_gui_qt_v3_final.py

Packaging (PyInstaller)

This folder contains a working spec: `TagOmatic-v4.0.spec`. To build locally:

   pip install pyinstaller
   pyinstaller TagOmatic-v3.1.spec

Notes

- Some models only support single images; the app adapts automatically
- IPTC/EXIF is written via ExifTool; see `exiftool_files/readme_windows.txt` for licensing info

Licensing overview

- PySide6 (Qt for Python): LGPLv3 (dynamically linked)
- Pillow: PIL license
- ExifTool: Artistic License 2.0
- Other dependencies retain their respective licenses

Support / Contact

- Project owner: Pistonspy / TagOmatic
- LinkedIn: https://www.linkedin.com/in/tonydewhurst/


