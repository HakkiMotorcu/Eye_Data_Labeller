"""Wrapper around micro_sam that the UI talks to.

Why a service class:
- Centralizes model loading (lazy, single source of truth).
- Hides the difference between "raw checkpoint on disk" and "named model
  variant from the micro-sam registry."
- Provides the seams the upcoming async + embedding-cache + prompt UX
  will plug into without further controller surgery.

Phase 4.1 scope (this file's first cut):
- ``available()``           — is micro_sam importable?
- ``load()``                — lazy model load with helpful error messages.
- ``auto_segment(frame)``   — run automatic instance segmentation on a
                              uint8 (H, W) or (H, W, 3) frame and return
                              an int instance-label array.

Phases 4.2+ will extend this with: precompute_embeddings, segment_from_box,
segment_from_points, and an async worker. The interface is intentionally
synchronous here — async is a higher-layer concern.
"""

import os
import numpy as np

try:
    import micro_sam
    from micro_sam import util
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


# Default checkpoint path inside the project — matches the collaborator
# training-output convention.
def default_sam_hela_path():
    """Resolve ``models/checkpoints/sam_hela/best.pt`` relative to the
    project root (the directory that contains ``main.py``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, 'models', 'checkpoints',
                        'sam_hela', 'best.pt')


class SamService:
    """Thin wrapper holding a loaded micro-sam predictor.

    Construct once per ``model_type`` + ``checkpoint_path`` combination
    and reuse — loading the model is the expensive step.
    """

    def __init__(self, model_type='vit_b', checkpoint_path=None, device=None):
        self.model_type = model_type
        # None means: let the user pick later or fall back to default_sam_hela_path().
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._predictor = None  # lazy

    # ---- Availability + load -----------------------------------------

    @staticmethod
    def available():
        return SAM_AVAILABLE

    def is_loaded(self):
        return self._predictor is not None

    def load(self):
        """Load the predictor if it isn't loaded yet.

        Raises FileNotFoundError if a checkpoint path was given but the
        file doesn't exist — caller decides how to surface that to the
        user (e.g. via QMessageBox).
        """
        if not SAM_AVAILABLE:
            raise RuntimeError(
                "micro_sam is not installed. Run `pip install micro_sam` "
                "into the project venv.")
        if self._predictor is not None:
            return self._predictor
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}")

        kwargs = dict(model_type=self.model_type)
        if self.checkpoint_path:
            kwargs['checkpoint_path'] = self.checkpoint_path
        if self.device:
            kwargs['device'] = self.device

        self._predictor = util.get_sam_model(**kwargs)
        return self._predictor

    def unload(self):
        """Drop the predictor so the user can switch models cleanly."""
        self._predictor = None

    # ---- Helpers ------------------------------------------------------

    @staticmethod
    def to_rgb_uint8(frame):
        """Coerce a (H, W) grayscale or (H, W, 3) RGB frame to uint8 RGB.

        micro_sam expects (H, W) or (H, W, 3) uint8. The existing display
        pipeline always normalizes to uint8 so this is usually a no-op.
        """
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    # ---- Inference: auto ---------------------------------------------

    def auto_segment(self, frame, verbose=False, **kwargs):
        """Automatic instance segmentation on a single 2D frame.

        Returns an int array of shape (H, W) where 0 is background and
        each connected component carries a unique label. Loads the model
        on first call.
        """
        self.load()
        frame_rgb = self.to_rgb_uint8(frame)
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=self.model_type, predictor=self._predictor)
        seg = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=frame_rgb,
            ndim=2,
            verbose=verbose,
            **kwargs,
        )
        return seg
