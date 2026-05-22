"""TIF mask I/O.

The on-disk canonical layout is a per-class instance-mask TIF inside the
project output folder, e.g. ``out/<video_stem>/Cells.tif``. Filenames and
the resolver live in :mod:`core.project_io`.

This module just owns the raw byte-level read/write of a single
``(T, H, W) uint16`` mask TIF and a project-folder sweep that loads
every class file at once. The next-to-video and bare-stem helpers that
older code shipped are gone.
"""

import os
import numpy as np
import tifffile


def _coerce_to_tyx(arr, axes, path_for_error):
    """Reshape arr so its layout becomes (T, Y, X), regardless of the
    axes label tifffile reports.

    tifffile sometimes returns axes='QYX' (Q = "unspecified") for masks
    written by Fiji macros — that's a stack we treat as T. Any singleton
    leading dims are squeezed; if more than one non-spatial dim remains
    we flatten them into T (preserves frame ordering).
    """
    if axes.endswith('YX'):
        leading = arr.shape[:-2]
        if not leading:
            return arr[np.newaxis, ...]
        # Flatten any combination of leading axes (T, Z, Q, ...) into one.
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    raise ValueError(
        f"Unsupported axes layout '{axes}' for {path_for_error}; "
        f"expected the last two axes to be Y, X.")


def load_mask_tif(path):
    """Load a mask TIF as ``(T, H, W) uint16``.

    Accepts single-page 2D, multi-page stack, or float-encoded integer
    masks (Fiji often writes float32 for integer-valued masks). Raises
    ``ValueError`` on shapes we don't understand.
    """
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = series.axes
        arr = series.asarray()

    # Drop a channel axis defensively — masks shouldn't have channels.
    if 'C' in axes:
        arr = np.take(arr, 0, axis=axes.index('C'))
        axes = axes.replace('C', '')

    arr = _coerce_to_tyx(arr, axes, path)

    # Round float-encoded masks (Fiji-style) before casting.
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)

    if arr.max() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Mask in {path} has instance IDs exceeding uint16 "
            f"({arr.max()} > 65535). Refusing to truncate.")
    if arr.min() < 0:
        raise ValueError(f"Mask in {path} has negative values.")

    return arr.astype(np.uint16, copy=False)


def save_mask_tif(masks, path):
    """Save ``(T, H, W)`` instance masks as a multi-page uint16 TIF.

    A 2D ``(H, W)`` array is auto-promoted to a single-frame stack so the
    output is always a multi-page file (one IFD per frame).

    Writes are atomic (temp file + rename) and an existing target is
    backed up to ``<path>.bak`` before being overwritten — see
    :func:`core.project_io.atomic_write_tif`.
    """
    from core import project_io
    project_io.atomic_write_tif(path, masks)


def load_multiclass_from_folder(folder):
    """Read per-class mask TIFs from a project output folder.

    Filenames follow the canonical names in
    ``project_io.CLASS_MASK_FILES``. Returns ``{class_type: arr}``;
    missing classes are omitted.
    """
    from core import project_io
    if not folder or not os.path.isdir(folder):
        return {}
    out = {}
    for ct, fname in project_io.CLASS_MASK_FILES.items():
        p = os.path.join(folder, fname)
        if os.path.exists(p):
            out[ct] = load_mask_tif(p)
    return out
