
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH)

runtime_dir = project_root / "runtime"
logo_file = project_root / "jero" / "ui" / "assets" / "JERO.png"
icon_file = project_root / "jero" / "ui" / "assets" / "JERO_Icon.ico"

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

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=runtime_binaries,
    datas=runtime_datas,
    hiddenimports=[],
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
    name="JERO i1",
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
    name="JERO i1",
)
