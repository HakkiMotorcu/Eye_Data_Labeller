"""Frame-source abstractions for the Eye Data Labeller.

The labeling UI talks to whatever provides:
  - ``num_frames`` (int)
  - ``height``, ``width`` (int)
  - ``filepath`` (str | None)
  - ``get_frame(i) -> np.ndarray (H, W)``

VideoData (core/volume_data.py) already satisfies this contract for
AVI/MP4/MOV. TiffFrameSource adds support for 2D and multi-page TIFFs.

Multi-channel TIFFs fall back to channel 0 with a console warning rather
than crashing. Float / uint32 inputs are linearly normalized to uint8 for
display compatibility with the existing pyqtgraph pipeline; raw data is
kept on the instance for future passes that need the original dtype.
"""

import os
from typing import Protocol, runtime_checkable
import numpy as np
import tifffile

VIDEO_EXTS = {'.avi', '.mp4', '.mkv', '.mov'}
TIFF_EXTS = {'.tif', '.tiff'}
SUPPORTED_EXTS = VIDEO_EXTS | TIFF_EXTS


def load_frame_source(path):
    """Open a file as the right kind of frame source based on extension.

    Lives here (not main.py) so File>Open / drag-drop / the session
    queue can dispatch without importing the entry-point module.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF_EXTS:
        return TiffFrameSource(path)
    if ext in VIDEO_EXTS:
        from core.volume_data import VideoData  # local: avoid cycle
        return VideoData(path)
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Accepted: {sorted(SUPPORTED_EXTS)}")


@runtime_checkable
class FrameSource(Protocol):
    """Duck-typed interface every image source must satisfy.

    Existing VideoData already conforms — nothing needs to subclass this;
    isinstance(obj, FrameSource) works via Protocol introspection.
    """

    filepath: object  # str | os.PathLike | None
    num_frames: int
    height: int
    width: int

    def get_frame(self, idx: int) -> np.ndarray: ...


class TiffFrameSource:
    """Load a 2D image or multi-page TIFF stack as (T, H, W) frames.

    Handles:
      * Single-page 2D TIFF → wrapped as a 1-frame stack
      * Multi-page stack with axes ``TYX`` / ``ZYX``
      * Multi-channel images / stacks (``YXC`` / ``TYXC``) → channel 0 + warn
      * Non-uint8 dtypes (float32, uint16, uint32) → linearly normalized
        to uint8 for display; original kept at ``self.frames_raw``
    """

    def __init__(self, filepath):
        self.filepath = filepath

        with tifffile.TiffFile(filepath) as tf:
            series = tf.series[0]
            axes = series.axes
            arr = series.asarray()

        # Drop channel axis if present.
        if 'C' in axes:
            c_idx = axes.index('C')
            n_channels = arr.shape[c_idx]
            print(f"TiffFrameSource: multi-channel TIFF (axes={axes}, "
                  f"{n_channels} channels) — using channel 0; multi-channel "
                  f"rendering is on the Phase 3 roadmap.")
            arr = np.take(arr, 0, axis=c_idx)
            axes = axes.replace('C', '')

        # Collapse any remaining axes to (T, Y, X). tifffile sometimes
        # reports 'QYX' (Q=unspecified) for Fiji-written stacks; treat
        # anything ending in YX as a (leading-dims-flattened, H, W) stack.
        if axes.endswith('YX'):
            if len(axes) == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        else:
            raise ValueError(
                f"Unsupported TIFF axes layout '{axes}' for {filepath}. "
                f"Expected the last two axes to be Y, X.")

        self.frames_raw = arr  # original dtype, kept for future SAM/analysis use
        self.frames = self._to_uint8(arr)
        self.num_frames, self.height, self.width = self.frames.shape

        print(f"TiffFrameSource loaded — {self.num_frames} frames, "
              f"{self.width}×{self.height}, raw dtype: {arr.dtype}, "
              f"display dtype: uint8")

    @staticmethod
    def _to_uint8(arr):
        if arr.dtype == np.uint8:
            return arr
        a = arr.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return np.zeros(a.shape, dtype=np.uint8)
        return ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)

    def get_frame(self, idx):
        idx = max(0, min(idx, self.num_frames - 1))
        return self.frames[idx]
