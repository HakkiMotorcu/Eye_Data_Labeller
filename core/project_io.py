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
                      keep_backup: bool = True,
                      compression: str = 'zlib',
                      compression_level: int = 6) -> None:
    """Write a (T, H, W) uint16 instance mask atomically.

    Instance masks are dominated by zero pixels and small repeating
    integer runs, so ZLIB (DEFLATE) typically shrinks them 10-100x with
    no quality loss. The output is still a standard TIF — Fiji,
    OpenCV, scikit-image, and pycocotools all read compressed TIFs
    transparently.

    Pass ``compression=None`` to write uncompressed (rare; useful only
    if a downstream tool insists).
    """
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
        kwargs = {}
        if compression:
            kwargs['compression'] = compression
            kwargs['compressionargs'] = {'level': int(compression_level)}
        tifffile.imwrite(tmp, arr, **kwargs)


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
    prev_status = None
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                prev = json.load(f) or {}
            created_at = prev.get('created_at', created_at)
            prev_status = prev.get('status')
        except (json.JSONDecodeError, OSError):
            pass  # corrupt manifest — overwrite cleanly

    manifest = {
        "version": PROJECT_SCHEMA_VERSION,
        "created_at": created_at,
        "updated_at": _now_iso(),
        # Explicit user-set work status — survives full rewrites.
        "status": prev_status,
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


def snapshot_existing_masks(out_folder: str, keep: int = 3):
    """Copy the folder's current mask TIFs + Meta.json into
    ``backup/session-<timestamp>/``.

    Called once per editing session before the first overwrite: the
    rolling ``.bak`` written by every save only survives until the
    NEXT save, so resuming an old session and saving twice used to
    silently destroy the resumed-from state. Keeps the newest ``keep``
    snapshots. Returns the snapshot dir, or None when there was
    nothing to back up.
    """
    import shutil
    if not out_folder or not os.path.isdir(out_folder):
        return None
    names = [*CLASS_MASK_FILES.values(), FILE_META]
    present = [n for n in names
               if os.path.exists(os.path.join(out_folder, n))]
    if not present:
        return None
    stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    dest = os.path.join(out_folder, 'backup', f'session-{stamp}')
    os.makedirs(dest, exist_ok=True)
    for n in present:
        shutil.copy2(os.path.join(out_folder, n), os.path.join(dest, n))
    root = os.path.join(out_folder, 'backup')
    snaps = sorted(d for d in os.listdir(root) if d.startswith('session-'))
    for d in snaps[:-keep]:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    return dest


# Explicit work status, set by the user at file-switch/close time
# ("Save & mark complete" / "Save & mark in progress"). Lives in
# project.json so it travels with the data.
STATUS_COMPLETE = 'complete'
STATUS_IN_PROGRESS = 'in_progress'


def read_status(out_folder: str) -> Optional[str]:
    """The manifest's explicit status field, or None."""
    m = read_project_manifest(out_folder)
    s = (m or {}).get('status')
    return s if s in (STATUS_COMPLETE, STATUS_IN_PROGRESS) else None


def write_status(out_folder: str, status: Optional[str]) -> None:
    """Update just the status field of project.json, creating a minimal
    manifest when none exists yet (bbox-only sessions have artifacts
    but no full save)."""
    ensure_dir(out_folder)
    path = os.path.join(out_folder, FILE_PROJECT)
    manifest = read_project_manifest(out_folder) or {
        "version": PROJECT_SCHEMA_VERSION,
        "created_at": _now_iso(),
    }
    manifest['status'] = status
    manifest['updated_at'] = _now_iso()
    atomic_write_json(path, manifest, keep_backup=False)


def file_status(out_folder: str) -> Optional[str]:
    """Display status for a stack's out folder.

    'complete' only when the user explicitly marked it so;
    'in_progress' when any session artifacts exist without that mark;
    None when untouched.
    """
    summary = session_summary(out_folder)
    if summary is None:
        return None
    if (summary.get('manifest') or {}).get('status') == STATUS_COMPLETE:
        return STATUS_COMPLETE
    return STATUS_IN_PROGRESS


# Class → export subfolder name for the by-class collate.
_CLASS_EXPORT_DIR = {
    'cell': 'Cells', 'vessel': 'Vessels', 'capillary': 'Capillaries',
}


def find_stack_folders(source_root: str):
    """Every folder under ``source_root`` that holds at least one class
    mask TIF — i.e. a per-stack output folder. Recursive, so it finds
    both ``<root>/<stem>/`` and the default ``<videos>/out/<stem>/``."""
    import shutil  # noqa: F401  (kept near collate for locality)
    found = []
    for dirpath, _dirs, files in os.walk(source_root):
        if any(f in files for f in CLASS_MASK_FILES.values()):
            found.append(dirpath)
    return sorted(found)


def collate_masks_by_class(source_root: str, dest_root: str):
    """Copy every stack's per-class masks into class folders:
    ``<dest>/Cells/<stem>.tif``, ``<dest>/Vessels/<stem>.tif``, … .

    Non-destructive (copies, never moves). Duplicate stem names get a
    numeric suffix so nothing is overwritten. Returns a summary dict:
    ``{'stacks': N, 'by_class': {'cell': n, ...}, 'collisions': [...]}``.
    """
    import shutil
    stacks = find_stack_folders(source_root)
    by_class = {ct: 0 for ct in CLASS_MASK_FILES}
    used = {ct: set() for ct in CLASS_MASK_FILES}
    collisions = []
    for folder in stacks:
        stem = os.path.basename(folder.rstrip(os.sep)) or 'stack'
        for ct, fname in CLASS_MASK_FILES.items():
            src = os.path.join(folder, fname)
            if not os.path.isfile(src):
                continue
            dest_dir = os.path.join(dest_root, _CLASS_EXPORT_DIR[ct])
            ensure_dir(dest_dir)
            name = stem
            n = 1
            while name in used[ct]:
                n += 1
                name = f"{stem}_{n}"
                if n == 2:
                    collisions.append(stem)
            used[ct].add(name)
            shutil.copy2(src, os.path.join(dest_dir, name + '.tif'))
            by_class[ct] += 1
    return {'stacks': len(stacks), 'by_class': by_class,
            'collisions': sorted(set(collisions))}


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
