"""TIF mask I/O matching the collaborator's labeling format.

The canonical on-disk label is a multi-page instance-mask TIF:
``{video}_Masks.tif`` of shape ``(T, H, W)``, where background = 0 and
each cell carries a unique uint16 instance ID.

Existing collaborator files are sometimes written as ``uint32`` or
``float32`` by Fiji's macro pipeline; the loader accepts any integer-
valued dtype and casts to uint16 (values up to 65 535 — more than enough
for the 5-35 cells per frame seen in PC-AOSLO data).
"""

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
    output is always a multi-page file (one IFD per frame), which is what
    the collaborator's Trackpy pipeline expects.

    Writes are atomic (temp file + rename) and an existing target is
    backed up to ``<path>.bak`` before being overwritten — see
    ``core.project_io.atomic_write_tif``.
    """
    from core import project_io
    project_io.atomic_write_tif(path, masks)


def default_mask_path(image_path):
    """Return the canonical sidecar mask path for a given image file.

    Mirrors the collaborator convention ``{stem}_Masks.tif``. Examples:
      308_20230822_02.avi → 308_20230822_02_Masks.tif
      2216_20231004_01.tif → 2216_20231004_01_Masks.tif
    """
    import os
    base, _ = os.path.splitext(image_path)
    return f"{base}_Masks.tif"


# ----- Per-class (3-layer) save/load --------------------------------------
# Suffix used for each class's mask TIF. Cells keep the legacy
# ``_Masks.tif`` suffix for backward compatibility so older sidecar files
# still match the load path.
CLASS_FILENAME_SUFFIX = {
    'cell':      '_Cells.tif',
    'vessel':    '_Vessels.tif',
    'capillary': '_Capillaries.tif',
}


def class_mask_path(image_path, class_type):
    """Return the per-class mask TIF path for a given image file."""
    import os
    base, _ = os.path.splitext(image_path)
    return f"{base}{CLASS_FILENAME_SUFFIX[class_type]}"


def save_multiclass_masks(seg, image_path):
    """Save all non-empty class layers as separate TIFs.

    Returns the list of paths actually written. Empty layers are skipped
    so we don't litter the directory with zero-content files.
    """
    written = []
    for ct, suffix in CLASS_FILENAME_SUFFIX.items():
        layer = seg.get_layer(ct)
        if layer is None or not layer.any():
            continue
        path = class_mask_path(image_path, ct)
        save_mask_tif(layer, path)
        written.append(path)
    return written


def load_multiclass_masks(image_path):
    """Look for per-class mask TIFs next to *image_path*.

    Returns a dict {class_type: (T, H, W) uint16}. Missing classes are
    omitted. If only the legacy ``_Masks.tif`` exists, it's returned
    under the ``cell`` key — older single-layer saves migrate cleanly.

    Also peeks into the new per-video output folder (``out/<stem>/``)
    when the per-stem siblings aren't found in the video's directory.
    """
    import os
    found = {}
    # Modern per-class files next to the source image (legacy layout).
    for ct in CLASS_FILENAME_SUFFIX:
        p = class_mask_path(image_path, ct)
        if os.path.exists(p):
            found[ct] = load_mask_tif(p)
    if 'cell' not in found:
        legacy = default_mask_path(image_path)
        if os.path.exists(legacy):
            found['cell'] = load_mask_tif(legacy)

    # If nothing matched, try the new out-folder layout.
    if not found:
        from core import project_io  # local import — avoid load-time cycle
        for mode in (project_io.OUTPUT_MODE_SUBFOLDER,
                     project_io.OUTPUT_MODE_PREFIXED):
            folder = project_io.resolve_output_folder(image_path, mode)
            extra = load_multiclass_from_folder(folder)
            if extra:
                found.update(extra)
                break
    return found


def load_multiclass_from_folder(folder):
    """Read per-class mask TIFs from a project output folder.

    Filenames follow the canonical names in
    ``project_io.CLASS_MASK_FILES``. Returns ``{class_type: arr}``;
    missing classes are omitted.
    """
    import os
    from core import project_io
    if not folder or not os.path.isdir(folder):
        return {}
    out = {}
    for ct, fname in project_io.CLASS_MASK_FILES.items():
        p = os.path.join(folder, fname)
        if os.path.exists(p):
            out[ct] = load_mask_tif(p)
    return out
