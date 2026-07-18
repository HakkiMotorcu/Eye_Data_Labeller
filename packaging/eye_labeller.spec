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

# ---- Qt platform plugins (headless "offscreen") -----------------------
# main.py --selftest (run by CI on the built bundle) forces
# QT_QPA_PLATFORM=offscreen so it can build real widgets with no display.
# PyInstaller's PyQt6 hook bundles the on-screen platform plugin
# (libqcocoa.dylib on macOS, qwindows.dll on Windows) but on some PyQt6
# layouts it does NOT collect the offscreen plugin, so the built app
# aborts at startup (SIGABRT / exit 134) with:
#   'Could not find the Qt platform plugin "offscreen" in
#    .../_internal/PyQt6/Qt6/plugins/platforms'
# Force-collect the offscreen plugin from the PyQt6 package into the
# platforms dir Qt scans (PyQt6/Qt6/plugins/platforms in the bundle).
# On Windows a missing offscreen plugin doesn't abort — Qt pops a modal
# "no Qt platform plugin could be initialized" dialog that hangs a
# headless run forever. Linux already passes because the hook collects
# it; the explicit add is a harmless de-dupe there.
#
# IMPORTANT: locate the plugin via the PyQt6 PACKAGE path only — do NOT
# `import PyQt6.QtCore` / use QLibraryInfo here. During the Windows build
# `import PyQt6.QtCore` fails (conda's ICU shadows PyQt6's — see main.py's
# runtime fix, which the spec process does NOT run), and that exception
# would skip this whole block, leaving the Windows bundle without the
# offscreen plugin. `import PyQt6` (the bare package) loads no Qt DLL and
# works everywhere.
try:
    import PyQt6
    _plat_dir = Path(PyQt6.__file__).parent / "Qt6" / "plugins" / "platforms"
    _offscreen = next(
        (_plat_dir / n
         for n in ("libqoffscreen.dylib", "qoffscreen.dll", "libqoffscreen.so")
         if (_plat_dir / n).is_file()),
        None,
    )
    if _offscreen is not None:
        binaries.append((str(_offscreen), "PyQt6/Qt6/plugins/platforms"))
except Exception:
    pass

# ---- Linux: bundle the env's libdeflate --------------------------------
# The Linux bundle ships libOpenEXRCore.so.33 (linked in via vigra, a
# micro_sam compute dep) but PyInstaller does not always collect the
# env's libdeflate beside it. At runtime the loader then resolves
# libdeflate.so.0 to the SYSTEM copy — Ubuntu 22.04 ships libdeflate
# 1.10, which predates the libdeflate_alloc_compressor_ex API OpenEXR
# needs — and micro_sam's import dies inside the bundle with:
#   libOpenEXRCore.so.33: undefined symbol: libdeflate_alloc_compressor_ex
# (sam_service swallows it, so SAM was silently dead on Linux.) In the
# live env it works because the loader finds conda's newer libdeflate.
# Same pattern as the pillow/cv2 fixes: force the matching copy in.
if sys.platform.startswith("linux"):
    import glob as _glob
    for _p in sorted(_glob.glob(os.path.join(sys.prefix, "lib",
                                             "libdeflate.so*"))):
        binaries.append((_p, "."))

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
    # matplotlib itself is handled per-platform just below (NOT
    # unconditionally excluded): micro_sam imports it from deeper
    # submodules (e.g. micro_sam.util, which core/sam_service.py loads),
    # so excluding it makes that import raise ModuleNotFoundError, and
    # sam_service catches it and silently disables SAM. matplotlib.tests
    # is always excluded — pure dead weight.
    "matplotlib.tests", "scipy.tests",
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

# matplotlib, per-platform. Keep it on macOS/Windows so micro_sam (and
# thus SAM) imports cleanly. EXCLUDE it on Linux, where bundling it is a
# net negative: pyqtgraph's ColorMapMenu eagerly imports matplotlib
# while constructing every pg.ImageView (find_mpl_leftovers ->
# colormap.listMaps), and matplotlib's bundled native stack has
# undefined symbols in the frozen Linux app — historically a conda
# libtiff ('undefined symbol: jpeg12_write_raw_data'), currently
# libraqm/harfbuzz ('undefined symbol: hb_ft_font_get_ft_face') — which
# crashes the app at ImageView construction. And micro_sam already fails
# to import in the Linux bundle for an unrelated native-symbol reason
# (libOpenEXRCore/libdeflate), so SAM is dead on Linux regardless;
# excluding matplotlib there costs no SAM support while keeping the app
# runnable. (See the offscreen-plugin block above for the other half of
# the macOS-arm64 selftest fix.)
if sys.platform.startswith("linux"):
    excludes.append("matplotlib")

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

# ---- macOS: do NOT bundle conda's libiconv ----------------------------
# Two libiconv flavors exist with the SAME install name: Apple's (in the
# dyld shared cache, exports _iconv/_iconv_open) and conda's GNU build
# (exports the prefixed _libiconv* symbols only). pip's cv2 wheel vendors
# a libintl.8.dylib that was delocate-built against APPLE's flavor; when
# the bundle also ships conda's libiconv.2.dylib, dyld binds cv2's
# libintl to it and dies at first (lazy) use:
#   dyld: Symbol not found: _iconv
#     Referenced from: _internal/cv2/.dylibs/libintl.8.dylib
#     Expected in:     _internal/libiconv.2.dylib
# This bind is lazy, so it fired nondeterministically — including AFTER
# 'selftest: PASS' was printed (the "flaky teardown segfault" on CI).
# Dropping conda's copy from the bundle makes dyld fall back to the
# system libiconv in the shared cache, which exports what cv2 expects.
# (macOS is guaranteed to provide it — it lives in the dyld cache on
# every supported macOS, so the bundle stays self-contained in practice.)
if sys.platform == "darwin":
    a.binaries = [b for b in a.binaries
                  if not os.path.basename(b[0]).startswith("libiconv")]

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
