"""Motion-related display enhancements: background subtraction.

Same contract as core/enhance.py — never mutates source data. Takes the
full stack + a center index and returns a (H, W) uint8 image showing
``|frame[i] - mean(frames[i-w : i+w+1])|`` scaled to uint8.

Useful for surfacing slow-moving immune cells inside a stationary vessel
in AOSLO videos — analogous to the std-projection mode but localized to
the current frame.
"""

import numpy as np


def bg_subtract(frames, center, window=2):
    """Subtract the rolling-mean background around the given frame.

    Returns a uint8 (H, W) image. ``window`` is the half-width — total
    span is ``2*window + 1`` frames clamped to the stack bounds.
    """
    if frames is None or frames.ndim != 3:
        raise ValueError("bg_subtract expects a (T, H, W) stack")
    T = frames.shape[0]
    if T <= 1:
        return frames[0].astype(np.uint8, copy=True)

    center = max(0, min(int(center), T - 1))
    window = max(1, int(window))
    lo = max(0, center - window)
    hi = min(T, center + window + 1)

    stack = frames[lo:hi].astype(np.float32)
    mean = stack.mean(axis=0)
    diff = np.abs(frames[center].astype(np.float32) - mean)
    mx = float(diff.max())
    if mx <= 0:
        return np.zeros(diff.shape, dtype=np.uint8)
    return (diff / mx * 255.0).astype(np.uint8)
