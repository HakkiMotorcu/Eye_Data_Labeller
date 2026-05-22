"""Path resolution that survives PyInstaller bundling.

When the app runs from source, ``project_root()`` returns the directory
containing ``main.py`` and everything just works. When PyInstaller
freezes the app, that directory becomes a read-only ``_MEIPASS`` temp
folder — fine for reading bundled resources, useless for writing
caches or downloaded model weights.

This module exposes two distinct roots:

* ``bundled_root()``   — where read-only resources live (icons, theme,
                          static assets we ship inside the bundle).
* ``user_data_root()`` — where writeable data lives across launches:
                          downloaded models, embedding cache, settings.

Use ``user_data_root() / "checkpoints" / "sam_hela" / "best.pt"`` for
the model path; ``bundled_root() / "icons" / "eye.png"`` for read-only
resources.
"""

from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def bundled_root() -> str:
    """Directory holding read-only resources shipped with the app.

    In a PyInstaller bundle this is ``sys._MEIPASS`` (the temp folder
    PyInstaller extracts into). From source it's the directory
    containing ``main.py`` (project root).
    """
    if is_frozen():
        return sys._MEIPASS  # type: ignore[attr-defined]
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _platform_user_data_root(app_name: str = "EyeDataLabeller") -> str:
    """OS-appropriate writeable per-user data folder."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(
            "~/AppData/Local")
    else:
        # XDG_DATA_HOME on Linux + most BSDs.
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser(
            "~/.local/share")
    return os.path.join(base, app_name)


def user_data_root() -> str:
    """Directory for per-user writable data.

    When NOT frozen, defaults to ``<project_root>/.user_data`` so dev
    runs keep their state next to the source tree. When frozen, uses
    the OS-standard per-user data location (so the bundled app doesn't
    try to write into its own read-only directory).
    """
    if is_frozen():
        root = _platform_user_data_root()
    else:
        root = os.path.join(bundled_root(), ".user_data")
    os.makedirs(root, exist_ok=True)
    return root


def default_sam_hela_checkpoint_path() -> str:
    """Canonical location for the auto-downloaded SAM-HeLa checkpoint.

    Always under ``user_data_root()`` so the file survives bundle
    upgrades and lives in a writeable directory regardless of how the
    app was launched.
    """
    return os.path.join(
        user_data_root(), "models", "checkpoints", "sam_hela", "best.pt")
