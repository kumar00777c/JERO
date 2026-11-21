# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH)

runtime_dir = project_root / "runtime"
logo_file = project_root / "jero" / "ui" / "assets" / "JERO.png"
icon_file = project_root / "jero" / "ui" / "assets" / "JERO_Icon.ico"
models_dir = project_root / "models" / "voice"

runtime_binaries = []
runtime_datas = []

# Include the local llama.cpp executable and its supporting DLLs.
for file in runtime_dir.iterdir():
    if file.is_file():
        if file.suffix.lower() == ".dll" or file.name.lower() == "llama-server.exe":
            runtime_binaries.append((str(file), "runtime"))

# Include the existing in-app logo without changing its location in the source project.
if logo_file.is_file():
    runtime_datas.append((str(logo_file), "jero/ui/assets"))

if icon_file.is_file():
    runtime_datas.append((str(icon_file), "jero/ui/assets"))

# ============================================================
# JERO i2 — Voice models (ONNX)
# ============================================================
# Bundled voice models are placed under models/voice/ in the project root.
# They are copied into the bundle at <MEIPASS>/models/voice so the
# VoiceConfig._models_dir() helper resolves them correctly at runtime.

if models_dir.is_dir():
    for model_file in models_dir.rglob("*"):
        if model_file.is_file():
            # Preserve the relative path under models/voice
            rel = model_file.relative_to(models_dir)
            runtime_datas.append((str(model_file), str(Path("models") / "voice" / rel)))

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=runtime_binaries,
    datas=runtime_datas,
    hiddenimports=[
        # Voice pipeline
        "jero.voice",
        "jero.voice.state_machine",
        "jero.voice.audio_input",
        "jero.voice.stt",
        "jero.voice.tts",
        "jero.voice.wake_detector",
        "jero.voice.worker",
        "jero.ui.voice_window",
        "jero.config",
        # ONNX Runtime (PyInstaller needs explicit help finding it)
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.pybind_state",
        # faster-whisper / CTranslate2
        "faster_whisper",
        "ctranslate2",
        # Piper TTS
        "piper",
        "piper.piper",
        "piper.config",
        # sounddevice (uses libportaudio)
        "sounddevice",
        # numpy / scipy
        "numpy",
        "scipy",
        "scipy.signal",
        # PySide6
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="JERO i2",
    icon=str(icon_file),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="JERO i2",
)