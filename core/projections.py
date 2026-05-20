"""Temporal projections across a (T, H, W) stack.

The headline mode for retinal AOSLO data is **std** — pixels whose intensity
changes a lot across frames have high temporal standard deviation. Stationary
vessel walls drop out, moving cells inside the vessels pop. Max / mean / sum
are also useful for navigating the dataset before labeling.

Projections are display-only: the raw frame stack is never mutated.
"""

import numpy as np


_MODES = ('none', 'std', 'max', 'mean', 'sum', 'min')


def available_modes():
    return _MODES


def project_stack(frames, mode, window=None, center=None):
    """Reduce a ``(T, H, W)`` stack to one ``(H, W)`` projection.

    ``mode``: one of available_modes(). ``'none'`` is a no-op pass-through.
    ``window`` + ``center``: if both set, projects over the ±window frames
    around ``center`` instead of the whole stack. Useful for "what's moving
    around frame i" workflows.

    Output is a uint8 image scaled to the data's own min/max.
    """
    if mode == 'none' or mode is None:
        return None
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]

    if window is not None and center is not None:
        lo = max(0, int(center) - int(window))
        hi = min(frames.shape[0], int(center) + int(window) + 1)
        frames = frames[lo:hi]

    f = frames.astype(np.float32, copy=False)
    if mode == 'std':
        out = f.std(axis=0)
    elif mode == 'max':
        out = f.max(axis=0)
    elif mode == 'mean':
        out = f.mean(axis=0)
    elif mode == 'sum':
        out = f.sum(axis=0)
    elif mode == 'min':
        out = f.min(axis=0)
    else:
        raise ValueError(f"Unknown projection mode: {mode!r}")

    mn, mx = float(out.min()), float(out.max())
    if mx <= mn:
        return np.zeros(out.shape, dtype=np.uint8)
    return ((out - mn) / (mx - mn) * 255.0).astype(np.uint8)
