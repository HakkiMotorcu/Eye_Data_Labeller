"""Temporal projections across a (T, H, W) stack.

The headline mode for retinal AOSLO data is **std** — pixels whose intensity
changes a lot across frames have high temporal standard deviation. Stationary
vessel walls drop out, moving cells inside the vessels pop. Max / mean / sum
are also useful for navigating the dataset before labeling.

Projections are display-only: the raw frame stack is never mutated.
"""

import numpy as np


_MODES = ('none', 'std', 'std_sum', 'max', 'mean', 'sum', 'min')


def available_modes():
    return _MODES


def _select_frames(frames, window_mode, center, window, range_lo, range_hi):
    """Slice ``frames`` along axis 0 according to the window mode.

    ``window_mode`` ∈ {'all', 'sliding', 'range'}.
    """
    T = frames.shape[0]
    if window_mode == 'sliding' and center is not None and window is not None:
        lo = max(0, int(center) - int(window))
        hi = min(T, int(center) + int(window) + 1)
        return frames[lo:hi]
    if window_mode == 'range' and range_lo is not None and range_hi is not None:
        lo = max(0, int(range_lo))
        hi = min(T, int(range_hi) + 1)
        if hi <= lo:
            hi = lo + 1
        return frames[lo:hi]
    return frames


def _std_sum_projection(f, chunk):
    """Sum-of-chunked-stddev: split the (T, H, W) array into non-
    overlapping windows of ``chunk`` frames, take the temporal stddev of
    each window, and sum the resulting maps.

    Highlights pixels that move *within short stretches* of the video —
    so a cell that briefly traverses a region contributes once, instead
    of being averaged out across the whole stack as plain std does.

    Falls back to plain std when there are fewer than ``chunk`` frames.
    """
    T = f.shape[0]
    chunk = max(2, int(chunk))
    if T < chunk:
        return f.std(axis=0)
    out = np.zeros(f.shape[1:], dtype=np.float32)
    for i in range(0, T - chunk + 1, chunk):  # non-overlapping windows
        out += f[i:i + chunk].std(axis=0)
    return out


def project_stack(frames, mode, *,
                  window_mode='all', window=None, center=None,
                  range_lo=None, range_hi=None,
                  percentile_clip=False, clip_lo=1.0, clip_hi=99.0,
                  std_sum_chunk=6):
    """Reduce a ``(T, H, W)`` stack to one ``(H, W)`` projection.

    ``mode``: one of available_modes(). ``'none'`` is a no-op pass-through.

    ``window_mode``:
      * ``'all'``     — project across every frame in the stack (default).
      * ``'sliding'`` — project across ``±window`` frames around ``center``.
                        Both must be supplied. The projection then follows
                        the active frame.
      * ``'range'``   — project across frames ``[range_lo, range_hi]``.

    ``percentile_clip``: when True, normalize using the [clip_lo, clip_hi]
    percentiles of the projection instead of its raw min/max. Stops a
    single hot pixel from washing out the displayed contrast.

    ``std_sum_chunk``: chunk size used by the ``'std_sum'`` mode. Ignored
    by all other modes.

    Output is a uint8 image.
    """
    if mode == 'none' or mode is None:
        return None
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]

    selected = _select_frames(frames, window_mode, center, window, range_lo, range_hi)

    f = selected.astype(np.float32, copy=False)
    if mode == 'std':
        out = f.std(axis=0)
    elif mode == 'std_sum':
        out = _std_sum_projection(f, std_sum_chunk)
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

    if percentile_clip:
        mn, mx = float(np.percentile(out, clip_lo)), float(np.percentile(out, clip_hi))
    else:
        mn, mx = float(out.min()), float(out.max())

    if mx <= mn:
        return np.zeros(out.shape, dtype=np.uint8)
    return np.clip((out - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
