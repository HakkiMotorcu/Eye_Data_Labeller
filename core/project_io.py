"""Project I/O layer for the Eye Data Labeller.

Responsibilities:

* **Output folder resolution** — given a source video path and an output
  mode, return the per-video output directory where this session's
  exports live.
* **Atomic file writes** — write to ``<path>.tmp`` then ``os.replace``
  onto the final name so a crash mid-write never leaves a half-file.
* **Backup on overwrite** — before clobbering an existing file, move it
  to ``<path>.bak``. Cheap insurance against accidental wipe-outs.
* **Project manifest** — a small ``project.json`` per output folder
  with source-video reference, schema versions, timestamps, and counts.

Output folder modes
-------------------

* ``"next_to_video_out_subfolder"`` (default) — ``<video_dir>/out/<stem>/``
* ``"next_to_video_prefixed"``                — ``<video_dir>/<stem>_out/``
* ``"custom_root"``                            — ``<custom_root>/<stem>/``

Settings persist via ``QSettings`` (cross-platform: registry on
Windows, plist on macOS, ini on Linux). The constants below match the
QSettings keys used elsewhere in the app.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from contextlib import contextmanager
from typing import Optional

import numpy as np
import tifffile


# ----- Settings keys (also used by the I/O Settings dialog) ----------------
SETTING_OUTPUT_MODE = "io/output_mode"
SETTING_OUTPUT_CUSTOM_ROOT = "io/output_custom_root"
SETTING_AUTOSAVE_MODE = "io/autosave_mode"
SETTING_AUTOSAVE_INTERVAL_SEC = "io/autosave_interval_sec"
SETTING_AUTOSAVE_MASK_MIN_SEC = "io/autosave_mask_min_sec"

# Output modes
OUTPUT_MODE_SUBFOLDER = "next_to_video_out_subfolder"
OUTPUT_MODE_PREFIXED  = "next_to_video_prefixed"
OUTPUT_MODE_CUSTOM    = "custom_root"

# Auto-save modes
AUTOSAVE_OFF   = "off"
AUTOSAVE_LIGHT = "light"   # annotations + meta only
AUTOSAVE_SMART = "smart"   # light + masks when dirty + N min elapsed

DEFAULTS = {
    SETTING_OUTPUT_MODE: OUTPUT_MODE_SUBFOLDER,
    SETTING_OUTPUT_CUSTOM_ROOT: "",
    SETTING_AUTOSAVE_MODE: AUTOSAVE_LIGHT,
    SETTING_AUTOSAVE_INTERVAL_SEC: 30,
    SETTING_AUTOSAVE_MASK_MIN_SEC: 300,
}

# Canonical filenames inside an output folder.
FILE_PROJECT       = "project.json"
FILE_META          = "Meta.json"
FILE_DRAFTS        = "Drafts.json"
FILE_AUTOSAVE      = "autosave.json"
CLASS_MASK_FILES   = {
    'cell':      'Cells.tif',
    'vessel':    'Vessels.tif',
    'capillary': 'Capillaries.tif',
}

PROJECT_SCHEMA_VERSION = 1


# ============================================================
#  Output-folder resolution
# ============================================================

def video_stem(video_path: str) -> str:
    """Filename without extension. Empty when path is empty."""
    if not video_path:
        return ""
    return os.path.splitext(os.path.basename(video_path))[0]


def resolve_output_folder(video_path: str, mode: str,
                           custom_root: str = "") -> str:
    """Return the per-video output folder for the given mode.

    Does NOT create the directory — callers do that lazily on first
    write (so opening a video doesn't litter the filesystem).
    """
    if not video_path:
        return ""
    stem = video_stem(video_path)
    video_dir = os.path.dirname(os.path.abspath(video_path))
    if mode == OUTPUT_MODE_PREFIXED:
        return os.path.join(video_dir, f"{stem}_out")
    if mode == OUTPUT_MODE_CUSTOM and custom_root:
        return os.path.join(os.path.expanduser(custom_root), stem)
    # Default fallback covers OUTPUT_MODE_SUBFOLDER and any unknown mode.
    return os.path.join(video_dir, "out", stem)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
#  Atomic + backup writes
# ============================================================

@contextmanager
def _backup_then_atomic(target_path: str, keep_backup: bool = True):
    """Yield a temp path; on success, atomically replace target.

    Steps:
      1. If target exists and ``keep_backup`` is True, copy it to
         ``target.bak`` (overwriting any previous backup).
      2. Yield a sibling temp path the caller writes to.
      3. On clean exit, ``os.replace(tmp, target)`` — atomic on POSIX
         and Windows when both paths are on the same filesystem.
      4. On exception, remove the temp file and leave the original
         (and its backup) untouched.
    """
    ensure_dir(os.path.dirname(os.path.abspath(target_path)) or ".")
    tmp = f"{target_path}.tmp"
    backup = f"{target_path}.bak"
    # Clean up any straggling temp from a previous crash.
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    try:
        yield tmp
    except Exception:
        # Wipe the partial temp; original is untouched.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if keep_backup and os.path.exists(target_path):
        try:
            shutil.copy2(target_path, backup)
        except OSError:
            # Backup failure shouldn't block the save — log via print
            # and continue (real callers can log_error themselves).
            print(f"project_io: backup of {target_path} failed; "
                  f"proceeding with overwrite anyway.")
    os.replace(tmp, target_path)


def atomic_write_json(path: str, obj, *, keep_backup: bool = True) -> None:
    """Write ``obj`` as deterministic JSON to ``path`` atomically."""
    with _backup_then_atomic(path, keep_backup=keep_backup) as tmp:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True)


def atomic_write_tif(path: str, arr: np.ndarray, *,
                      keep_backup: bool = True) -> None:
    """Write a (T, H, W) uint16 instance mask atomically."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"atomic_write_tif expects (T,H,W) or (H,W); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    if arr.max() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Cannot save mask: max instance ID {arr.max()} exceeds uint16.")
    arr = arr.astype(np.uint16, copy=False)
    with _backup_then_atomic(path, keep_backup=keep_backup) as tmp:
        tifffile.imwrite(tmp, arr)


# ============================================================
#  Project manifest
# ============================================================

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def write_project_manifest(out_folder: str, *,
                           source_video_path: str,
                           frame_count: int,
                           frame_size: tuple[int, int],
                           class_counts: dict[str, int],
                           extra: Optional[dict] = None) -> str:
    """Write / update ``project.json`` in the output folder.

    ``created_at`` is preserved across rewrites; ``updated_at`` always
    refreshes. Returns the project file path.
    """
    ensure_dir(out_folder)
    path = os.path.join(out_folder, FILE_PROJECT)
    created_at = _now_iso()
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                prev = json.load(f) or {}
            created_at = prev.get('created_at', created_at)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt manifest — overwrite cleanly

    manifest = {
        "version": PROJECT_SCHEMA_VERSION,
        "created_at": created_at,
        "updated_at": _now_iso(),
        "source_video": {
            "absolute_path": os.path.abspath(source_video_path)
                              if source_video_path else "",
            "filename": os.path.basename(source_video_path)
                          if source_video_path else "",
        },
        "frame_count": int(frame_count),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "exports": {
            "cells":       CLASS_MASK_FILES['cell'],
            "vessels":     CLASS_MASK_FILES['vessel'],
            "capillaries": CLASS_MASK_FILES['capillary'],
            "meta":        FILE_META,
        },
    }
    if extra:
        manifest.update(extra)
    atomic_write_json(path, manifest, keep_backup=False)
    return path


def read_project_manifest(out_folder: str) -> Optional[dict]:
    """Return parsed project.json or None if absent / unreadable."""
    path = os.path.join(out_folder, FILE_PROJECT)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================
#  Discovery — does this video already have a saved session?
# ============================================================

def session_summary(out_folder: str) -> Optional[dict]:
    """Return a short summary of what's already in the out folder.

    None when the folder doesn't exist or is empty. Otherwise a dict
    like::

        {
          "out_folder": ".../out/<stem>",
          "manifest":   <project.json or None>,
          "files":      ["Cells.tif", "Meta.json", ...],
          "updated_at": "2026-...",  # newest mtime among known files
          "has_masks":  True,
        }

    Used by the resume prompt on video open.
    """
    if not out_folder or not os.path.isdir(out_folder):
        return None
    known = [FILE_PROJECT, FILE_META, FILE_DRAFTS, FILE_AUTOSAVE,
             *CLASS_MASK_FILES.values()]
    present = [name for name in known
               if os.path.exists(os.path.join(out_folder, name))]
    if not present:
        return None
    newest = max(
        os.path.getmtime(os.path.join(out_folder, name)) for name in present)
    has_masks = any(
        os.path.exists(os.path.join(out_folder, CLASS_MASK_FILES[c]))
        for c in CLASS_MASK_FILES)
    manifest = read_project_manifest(out_folder)
    return {
        "out_folder": out_folder,
        "manifest": manifest,
        "files": present,
        "updated_at": _dt.datetime.fromtimestamp(newest).isoformat(
            timespec='seconds'),
        "has_masks": has_masks,
    }
