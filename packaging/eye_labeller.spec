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
    collect_data_files, collect_dynamic_libs, collect_submodules,
    copy_metadata,
)
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).parent      # noqa: F821  (SPECPATH provided by PyInstaller)
APP_NAME = "EyeDataLabeller"
ENTRY = str(PROJECT_ROOT / "main.py")

# ---- Hidden imports ---------------------------------------------------
# PyTorch / micro-sam / trackastra dynamically import submodules that
# PyInstaller can't trace statically — collect them up front.
#
# qdarktheme: main.py imports `qdarktheme` inside try/except. PyInstaller
# DOES trace the import, but the package ships .qss/.svg theme assets
# that only get bundled via collect_data_files (below). Without those,
# load_stylesheet("dark") raises and we silently fall back to system
# theme — listing it here for symmetry with the data-files block.
#
# cv2 (opencv): imported at module top in core/volume_data.py so the
# static scanner picks up the main module, but opencv's codec plugins
# (libjpeg/libpng/libtiff backends, video backends) live as separate
# dylibs that PyInstaller often misses — collect_dynamic_libs catches
# them. Without this, cv2.imread / imwrite of certain formats raise
# cryptic "could not find a writer for the specified extension" errors.
hidden = []
for pkg in (
    "torch", "torchvision", "torch_em",
    "segment_anything", "micro_sam",
    "trackastra", "motile",
    "skimage", "scipy", "tifffile",
    "pyqtgraph", "qtawesome",
    "qdarktheme", "cv2",
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        # Optional deps: skip silently if not installed in the build env.
        pass

# ---- Data files -------------------------------------------------------
# Anything the apps reads at runtime that isn't a .py file: model
# configs shipped by micro_sam, pyqtgraph icon resources, qtawesome
# font files, qdarktheme stylesheets, etc.
datas = []
for pkg in (
    "micro_sam", "segment_anything", "trackastra",
    "pyqtgraph", "qtawesome", "qdarktheme",
):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ---- Dynamic libraries ------------------------------------------------
# Some packages ship plugin-style .dylib/.so files alongside their main
# binary — PyInstaller's hook system catches most, but cv2's image-
# codec backends and ffmpeg shim are inconsistent across opencv builds.
# Force-collect to avoid "could not find a writer" failures at runtime.
binaries = []
for pkg in ("cv2",):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# cv2 also stashes its native module at
# cv2/python-<ver>/cv2.cpython-<ver>-<plat>.so. The Python-ABI-tagged
# filename trips collect_dynamic_libs's *.so / *.dylib glob in some
# PyInstaller versions, so the .so silently doesn't make it into the
# bundle and cv2 import hits a recursion ("ERROR: recursion is
# detected during loading of cv2 binary extensions"). Explicitly add
# every python-X.Y/ subdir as a data tree so the native .so lands at
# Contents/Frameworks/cv2/python-X.Y/ where the patched config-X.Y.py
# expects it.
try:
    import cv2 as _cv2
    import glob as _glob
    _cv2_dir = os.path.dirname(_cv2.__file__)
    for _pydir in _glob.glob(os.path.join(_cv2_dir, 'python-*')):
        if os.path.isdir(_pydir):
            datas.append((_pydir,
                          os.path.join('cv2', os.path.basename(_pydir))))
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
#
# PyQt5 + PySide2 + PySide6 — the conda env pulls PyQt5 in
# transitively (via qtpy / magicgui / napari-adjacent helpers shipped
# with micro_sam). PyInstaller refuses to bundle multiple Qt bindings
# in one app — without the exclude the build dies with:
#   "attempt to collect multiple Qt bindings packages"
# We're a PyQt6 app, so any PyQt5 reachable through `import qtpy` etc.
# at runtime is dead code for us.
excludes = [
    "PyQt5", "PySide2", "PySide6",
    # matplotlib: the app never imports it — it only sneaks in because
    # pyqtgraph's colormap menu lazily loads it WHEN PRESENT. Bundled,
    # it dragged in a conda libtiff whose 12-bit libjpeg symbols were
    # missing at runtime ('undefined symbol: jpeg12_write_raw_data'),
    # crashing ImageView construction. Excluded, pyqtgraph cleanly
    # skips matplotlib colormaps (the app's get_colormap already
    # guards its optional matplotlib fallback), and the bundle loses
    # ~60 MB of dead weight.
    "matplotlib", "matplotlib.tests", "scipy.tests",
    "tornado", "notebook", "jupyter", "jupyterlab",
    "IPython", "pytest",
    # Unused heavy GUI/vis stacks that ride in transitively (napari /
    # micro_sam pull them) and break the bundle: vispy tries to
    # dlopen system libfontconfig at import, napari/magicgui drag in
    # more Qt. The app imports none of them — exclude so PyInstaller
    # doesn't bundle (and mis-link) them.
    "vispy", "napari", "magicgui", "superqt", "qtpy",
    "PyQt5", "PySide2",
]

# ---- Analysis ---------------------------------------------------------
a = Analysis(
    [ENTRY],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
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
