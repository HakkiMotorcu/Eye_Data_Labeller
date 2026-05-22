# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Eye Data Labeller.

Build locally:

    cd <project_root>
    pyinstaller packaging/eye_labeller.spec --clean --noconfirm

Build via GitHub Actions: see .github/workflows/build.yml. The CI
workflow invokes this same spec on macOS, Windows, and Linux runners
to produce platform-native artifacts.

Tip: PyTorch + micro_sam are large and have unusual import patterns.
We collect them explicitly via `collect_submodules` / `collect_data_files`
so the bundle doesn't drop dynamic imports.
"""

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, copy_metadata,
)
import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).parent      # noqa: F821  (SPECPATH provided by PyInstaller)
APP_NAME = "EyeDataLabeller"
ENTRY = str(PROJECT_ROOT / "main.py")

# ---- Hidden imports ---------------------------------------------------
# PyTorch / micro-sam / trackastra dynamically import submodules that
# PyInstaller can't trace statically — collect them up front.
hidden = []
for pkg in (
    "torch", "torchvision", "torch_em",
    "segment_anything", "micro_sam",
    "trackastra", "motile",
    "skimage", "scipy", "tifffile",
    "pyqtgraph", "qtawesome",
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        # Optional deps: skip silently if not installed in the build env.
        pass

# ---- Data files -------------------------------------------------------
# Anything the apps reads at runtime that isn't a .py file: model
# configs shipped by micro_sam, pyqtgraph icon resources, qtawesome
# font files, etc.
datas = []
for pkg in (
    "micro_sam", "segment_anything", "trackastra",
    "pyqtgraph", "qtawesome",
):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# Some packages stash version info / metadata files PyInstaller doesn't
# pick up by default; copy them so importlib.metadata works at runtime.
for pkg in ("torch", "torchvision", "micro_sam", "trackastra",
            "segment_anything"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Our own files that aren't Python modules — currently nothing, but
# this is the place to drop assets if we add icons / theme files.
# datas += [(str(PROJECT_ROOT / "assets" / "*"), "assets")]

# ---- Excludes ---------------------------------------------------------
# Trim the bundle by excluding things we definitely don't use.
excludes = [
    "matplotlib.tests", "scipy.tests",
    "tornado", "notebook", "jupyter", "jupyterlab",
    "IPython", "pytest",
]

# ---- Analysis ---------------------------------------------------------
a = Analysis(
    [ENTRY],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ---- One-folder build (faster startup) -------------------------------
# We use the one-folder layout (collect into a directory) rather than
# one-file: avoids the ~10-30s extraction delay on every launch.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,    # GUI app on Win/Mac; no console window
    disable_windowed_traceback=False,
    icon=None,        # TODO: drop a per-OS icon into packaging/icons/ and reference it here
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

# macOS .app bundle wrapper.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="org.eyedatalabeller.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
