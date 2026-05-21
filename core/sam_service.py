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

from core.debug import log, log_error

try:
    import micro_sam
    from micro_sam import util
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )
    SAM_AVAILABLE = True
except ImportError as _exc:
    log_error('sam_service', 'micro_sam import failed', exc=_exc)
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
        self._state = None      # decoder_state + image_encoder weights — needed
                                # by get_predictor_and_segmenter for AIS mode.

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
            log('sam_service', 'load: already loaded', model_type=self.model_type)
            return self._predictor
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}")

        kwargs = dict(model_type=self.model_type)
        if self.checkpoint_path:
            kwargs['checkpoint_path'] = self.checkpoint_path
        if self.device:
            kwargs['device'] = self.device

        # return_state=True is required so that the automatic instance
        # segmenter has access to decoder_state (it asserts on `state`
        # being not None when given a pre-loaded predictor).
        kwargs['return_state'] = True
        log('sam_service', 'loading predictor + state', **kwargs)
        try:
            self._predictor, self._state = util.get_sam_model(**kwargs)
        except Exception as e:
            log_error('sam_service', 'get_sam_model failed', exc=e, **kwargs)
            raise
        has_decoder = isinstance(self._state, dict) and 'decoder_state' in self._state
        log('sam_service', 'predictor loaded',
            state_keys=list(self._state.keys()) if isinstance(self._state, dict) else type(self._state).__name__,
            has_decoder=has_decoder)
        return self._predictor

    def unload(self):
        """Drop the predictor + state so the user can switch models cleanly."""
        self._predictor = None
        self._state = None

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

    # ---- Inference: prompt-based ------------------------------------

    def segment_from_box(self, frame, box, *, multimask_output=False):
        """Run SAM with a single bbox prompt.

        Args:
          frame: (H, W) or (H, W, 3) uint8 — same SAM rule as auto_segment.
          box: (x0, y0, x1, y1) XYXY in image coordinates — what the rest
            of the app uses. This wrapper internally re-orders to the
            (y0, x0, y1, x1) layout micro_sam._process_box actually
            consumes (verify by reading
            micro_sam.prompt_based_segmentation._process_box: it indexes
            box[1] for x_min, box[0] for y_min, etc.).
          multimask_output: when True, SAM evaluates 3 candidate masks
            internally and returns the best-IoU one (still (H, W)).
            Often improves quality on ambiguous bboxes at the cost of a
            small extra inference pass. When False, returns a single
            mask from the primary prediction head.

        Returns:
          (H, W) bool mask.
        """
        from micro_sam.prompt_based_segmentation import segment_from_box as _sfb

        self.load()
        frame_rgb = self.to_rgb_uint8(frame)
        x0, y0, x1, y1 = (float(v) for v in box)
        # Re-pack to (y0, x0, y1, x1) for micro_sam.
        box_arr = np.array([y0, x0, y1, x1], dtype=np.float32)
        log('sam_service', 'segment_from_box: starting',
            frame_shape=frame_rgb.shape,
            box_xyxy=[x0, y0, x1, y1],
            box_passed_to_sam_yxyx=box_arr.tolist(),
            multimask=multimask_output)

        # micro_sam.segment_from_box reads the predictor's currently-set
        # image. We have to seed it before calling.
        try:
            self._predictor.set_image(frame_rgb)
            if multimask_output:
                # Ask for all 3 candidates + their predicted IoU scores
                # so we can pick the highest-scoring one ourselves.
                mask, scores, _logits = _sfb(
                    predictor=self._predictor,
                    box=box_arr,
                    multimask_output=True,
                    return_all=True,
                )
                mask = np.asarray(mask)              # shape (3, H, W)
                scores = np.asarray(scores).ravel()  # shape (3,)
                best = int(np.argmax(scores))
                log('sam_service', 'segment_from_box: picked best of 3',
                    scores=[round(float(s), 3) for s in scores], best_idx=best)
                mask = mask[best]
            else:
                mask = _sfb(
                    predictor=self._predictor,
                    box=box_arr,
                    multimask_output=False,
                    return_all=False,
                )
        except Exception as e:
            log_error('sam_service', 'segment_from_box raised',
                      exc=e, box=box_arr.tolist())
            raise
        # SAM returns (1, H, W) when not multimask; squeeze to (H, W).
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        log('sam_service', 'segment_from_box: done',
            result_shape=tuple(mask.shape), n_true=int(mask.sum()))
        return mask

    # ---- Inference: auto ---------------------------------------------

    def auto_segment(self, frame, verbose=False, **kwargs):
        """Automatic instance segmentation on a single 2D frame.

        Returns an int array of shape (H, W) where 0 is background and
        each connected component carries a unique label. Loads the model
        on first call.
        """
        self.load()
        frame_rgb = self.to_rgb_uint8(frame)
        log('sam_service', 'auto_segment: starting',
            frame_shape=frame_rgb.shape, dtype=str(frame_rgb.dtype),
            model_type=self.model_type)
        try:
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=self.model_type,
                predictor=self._predictor,
                state=self._state,
            )
            log('sam_service', 'auto_segment: got predictor + segmenter',
                segmenter_type=type(segmenter).__name__)
            seg = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=frame_rgb,
                ndim=2,
                verbose=verbose,
                **kwargs,
            )
        except Exception as e:
            log_error('sam_service', 'auto_segment failed', exc=e,
                      frame_shape=frame_rgb.shape, model_type=self.model_type)
            raise
        try:
            n_inst = int(np.unique(seg).size - 1)
        except Exception:
            n_inst = -1
        log('sam_service', 'auto_segment: done',
            result_shape=seg.shape, dtype=str(seg.dtype), n_instances=n_inst)
        return seg
