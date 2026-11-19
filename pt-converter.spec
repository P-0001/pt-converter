# -*- mode: python ; coding: utf-8 -*-
# pt-converter.spec — PyInstaller build spec for pt-converter
#
# Produces a single self-contained executable (one-file mode) for Windows x64
# (dist/pt-converter.exe), Linux x64 and macOS arm64 (dist/pt-converter, no
# extension). Build natively on each platform; PyInstaller cannot
# cross-compile. The entry point is pt-converter.py, which has an
# `if __name__ == "__main__":` guard.
#
# Expected output size: ~500 MB to >1 GB depending on the pinned Torch
# version and packaging. PyInstaller one-file mode extracts the bundle
# to a temp directory on every launch, so cold-start time is several
# seconds (often 5-15s) before the converter can do any work. Document
# this in the release notes; if reliability/cost is worse than the
# convenience of one download, the fallback is a signed one-directory
# build shipped as a ZIP (the Go orchestration contract is unchanged).
#
# NOTE on hidden imports: the list below covers the modules Ultralytics
# and PyTorch typically pull in lazily (importlib / dynamic registries).
# It may need adjustment after the first real build attempt — run the
# built executable against a fixture .pt model and add any missing modules
# PyInstaller reports as ModuleNotFoundError / ImportError.

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# PyInstaller's dependency analyzer recurses deeply through torch/ultralytics.
# The default recursion limit (1000) is not enough and causes RecursionError
# during the Analysis phase. Multiply it before anything else runs.
import sys as _sys
import os as _os

_sys.setrecursionlimit(_sys.getrecursionlimit() * 5)

# Application icon embedded into the exe's PE resources (Windows only).
# Resolved relative to the spec file (PyInstaller runs with cwd = spec dir).
# Falls back to no icon if the file is missing so the build still succeeds
# during local dev. On Linux and macOS the .ico is deliberately NOT passed:
# a one-file console binary has no icon resource there, and a .ico on macOS
# can trigger a Pillow icon-conversion step (.ico -> .icns) that may fail the
# build. Windows behavior is unchanged.
_icon_path = _os.path.join(SPECPATH, "assets", "pt-converter.ico")
_icon = _icon_path if _sys.platform == "win32" and _os.path.isfile(_icon_path) else None

block_cipher = None

# Collect Ultralytics' dynamic submodules and config/data files
# (model YAMLs, fonts, etc.) that PyInstaller cannot always detect.
ultralytics_hiddenimports = collect_submodules("ultralytics")
ultralytics_datas = collect_data_files("ultralytics")

hiddenimports = [
    # Core scientific / IO stack
    "numpy",
    "yaml",
    "cv2",
    "onnx",
    # PyTorch
    "torch",
    "torchvision",
    # Multiprocessing — torch uses it internally; PyInstaller may miss it.
    # (freeze_support is a function in this module, not a submodule, so it
    # is not listed separately.)
    "multiprocessing",
    # Ultralytics top-level packages (collect_submodules fills the rest)
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.engine",
    "ultralytics.models",
    "ultralytics.cfg",
] + ultralytics_hiddenimports

# Ultralytics pulls matplotlib, pandas, seaborn and psutil as dependencies
# for training/validation/plotting. The converter only does export, so we
# exclude the heavy plotting/data deps to shrink the bundle. If a missing
# module error appears at runtime, move the offending name into
# hiddenimports above instead of widening the excludes.
excludes = [
    "matplotlib",
    "pandas",
    "seaborn",
    "psutil",
    "IPython",
    "jupyter",
    "notebook",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
]

datas = ultralytics_datas

a = Analysis(
    ["pt-converter.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="pt-converter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX off by default — it triggers antivirus false positives.
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI tool: keep a console window for stdout/stderr.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Native arch of the build Python (x64 Windows/Linux, arm64 macOS).
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,  # None off Windows or when assets/pt-converter.ico is absent.
)
