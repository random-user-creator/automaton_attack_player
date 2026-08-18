from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


project_root = Path(os.environ["AUTOMATON_PROJECT_ROOT"])
model_root = Path(os.environ["AUTOMATON_MODEL_ROOT"])
src_root = project_root / "src"

datas = []
binaries = []
hiddenimports = []

for package in ("paddle", "paddleocr", "paddlex"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for distribution in (
    "imagesize",
    "modelscope",
    "opencv-contrib-python",
    "paddleocr",
    "paddlepaddle",
    "paddlex",
    "pyclipper",
    "pypdfium2",
    "python-bidi",
    "shapely",
):
    datas += copy_metadata(distribution, recursive=True)

for model_name in ("PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"):
    model_dir = model_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Required release model is missing: {model_dir}")
    datas.append((str(model_dir), f"models/{model_name}"))

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "easyocr",
        "onnxruntime",
        "pytesseract",
        "rapidocr",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AutomatonAttackPlayer",
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
