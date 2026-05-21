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

import hashlib
import os
from collections import OrderedDict

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

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


class EmbeddingPrecomputeWorker(QThread):
    """Background thread that precomputes embeddings for a list of frames.

    Only one worker should run at a time — SAM's predictor is not
    thread-safe. The ToolController serializes this.

    Signals:
      progress(done, total)
      frame_done(frame_idx)
      error(message)
      finished_ok()
    """
    progress = pyqtSignal(int, int)
    frame_done = pyqtSignal(int)
    error = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, sam_service, image_path, frames, parent=None):
        super().__init__(parent)
        self.sam_service = sam_service
        self.image_path = image_path
        # frames: iterable of (frame_idx, np.ndarray) pairs.
        self.frames = list(frames)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        total = len(self.frames)
        for i, (frame_idx, frame) in enumerate(self.frames):
            if self._stop_requested:
                log('embed_worker', 'stopped early', after=i, total=total)
                return
            try:
                self.sam_service.precompute_embedding(
                    frame, image_path=self.image_path, frame_idx=frame_idx)
            except Exception as e:
                log_error('embed_worker', 'precompute failed',
                          exc=e, frame_idx=frame_idx)
                self.error.emit(f"frame {frame_idx}: {type(e).__name__}: {e}")
                return
            self.frame_done.emit(int(frame_idx))
            self.progress.emit(i + 1, total)
        self.finished_ok.emit()


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

    # ---- Embedding cache defaults -----------------------------------
    _DEFAULT_MAX_RAM_EMBEDDINGS = 16  # ~16 * 4 MB = 64 MB RAM ceiling
    _DEFAULT_CACHE_ROOT = "~/.cache/eye_labeller/sam_embeddings"

    def __init__(self, model_type='vit_b', checkpoint_path=None, device=None,
                 max_ram_embeddings=None, cache_root=None):
        self.model_type = model_type
        # None means: let the user pick later or fall back to default_sam_hela_path().
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._predictor = None  # lazy
        self._state = None      # decoder_state + image_encoder weights — needed
                                # by get_predictor_and_segmenter for AIS mode.
        # ---- Embedding cache (Phase 1a) -----------------------------
        # Three tiers checked in order:
        #   1. RAM LRU (max_ram_embeddings entries)
        #   2. Disk zarr at cache_root/<image_hash>/<model_tag>/frame_NNNNN.zarr
        #   3. Recompute via micro_sam.util.precompute_image_embeddings
        self._max_ram_embeddings = (max_ram_embeddings
                                    or self._DEFAULT_MAX_RAM_EMBEDDINGS)
        self._cache_root = os.path.expanduser(
            cache_root or self._DEFAULT_CACHE_ROOT)
        self._embeddings = OrderedDict()  # LRU: cache_key -> ImageEmbeddings

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
        # RAM cache holds model-specific embeddings; clearing is correct
        # when the underlying model is gone. Disk cache stays (it's keyed
        # by model variant so coexists with other models).
        self.clear_ram_cache()

    # ---- Embedding cache helpers -----------------------------------

    def clear_ram_cache(self):
        """Drop all in-memory embeddings (used on model swap)."""
        n = len(self._embeddings)
        self._embeddings.clear()
        if n:
            log('sam_service', 'RAM cache cleared', n_evicted=n)

    def cache_dir_for_image(self, image_path):
        """Return the per-image, per-model cache directory.

        Layout:
            <cache_root>/<image_stem>_<hash10>/<model_tag>/
        Different model variants (e.g. sam_hela vs vit_b_lm) live in
        separate subfolders so switching doesn't invalidate either.
        """
        if not image_path:
            return None
        abs_path = os.path.abspath(image_path)
        h = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        if self.checkpoint_path:
            variant = os.path.basename(os.path.dirname(self.checkpoint_path))
            model_tag = f"{self.model_type}-{variant}"
        else:
            model_tag = self.model_type
        safe_tag = ''.join(
            c if c.isalnum() or c in '-_.' else '_' for c in model_tag)
        return os.path.join(self._cache_root, f"{stem}_{h}", safe_tag)

    def _cache_key(self, image_path, frame_idx):
        """RAM cache key — includes model identity so switching invalidates."""
        return (image_path, self.model_type, self.checkpoint_path, int(frame_idx))

    def _ram_store(self, key, emb):
        """Add to RAM LRU and evict oldest if over capacity."""
        self._embeddings[key] = emb
        while len(self._embeddings) > self._max_ram_embeddings:
            evicted, _ = self._embeddings.popitem(last=False)
            log('sam_service', 'embedding LRU-evicted from RAM',
                evicted_image=evicted[0], frame_idx=evicted[3])

    def has_cached_embedding(self, image_path, frame_idx):
        """True iff the embedding is already in RAM or on disk."""
        key = self._cache_key(image_path, frame_idx)
        if key in self._embeddings:
            return True
        cache_dir = self.cache_dir_for_image(image_path)
        if cache_dir is None:
            return False
        return os.path.exists(
            os.path.join(cache_dir, f"frame_{int(frame_idx):05d}.zarr"))

    def precompute_embedding(self, frame, *, image_path=None, frame_idx=None,
                              persist_to_disk=True):
        """Get-or-compute embedding for one frame.

        Lookup order: RAM (LRU) -> disk zarr -> compute fresh.
        Computed embeddings are written to RAM and (unless
        persist_to_disk=False) to disk.

        Without image_path or frame_idx, runs fresh every call with no
        caching — useful for one-off prompts on synthetic data.
        """
        from micro_sam import util as msu

        has_key = image_path is not None and frame_idx is not None
        key = self._cache_key(image_path, frame_idx) if has_key else None

        # 1. RAM hit
        if key is not None and key in self._embeddings:
            self._embeddings.move_to_end(key)
            log('sam_service', 'embedding RAM hit', frame_idx=frame_idx)
            return self._embeddings[key]

        self.load()  # ensure predictor

        # 2. Decide save_path. If the file already exists,
        # precompute_image_embeddings loads from it instead of recomputing.
        save_path = None
        if has_key and persist_to_disk:
            cache_dir = self.cache_dir_for_image(image_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                save_path = os.path.join(
                    cache_dir, f"frame_{int(frame_idx):05d}.zarr")

        disk_hit = save_path is not None and os.path.exists(save_path)
        log('sam_service',
            'embedding disk hit' if disk_hit else 'embedding compute',
            frame_idx=frame_idx, save_path=save_path)

        frame_rgb = self.to_rgb_uint8(frame)
        try:
            # ndim=2 is critical: without it micro_sam auto-detects (H, W, 3)
            # as a 3D (D=3) stack and tries to batch-encode three slices,
            # which fails on _compute_3d's batched torch.cat path.
            emb = msu.precompute_image_embeddings(
                self._predictor, frame_rgb,
                save_path=save_path, ndim=2, verbose=False,
            )
        except Exception as e:
            log_error('sam_service', 'precompute_image_embeddings failed',
                      exc=e, frame_idx=frame_idx)
            raise

        if key is not None:
            self._ram_store(key, emb)
        return emb

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

    def segment_from_box(self, frame, box, *,
                         image_path=None, frame_idx=None,
                         multimask_output=False):
        """Run SAM with a single bbox prompt.

        When ``image_path`` and ``frame_idx`` are provided, the image
        embedding is looked up in the RAM/disk cache and only computed
        if absent — subsequent calls on the same frame are essentially
        free (~50 ms vs ~2 s).

        Args:
          frame: (H, W) or (H, W, 3) uint8.
          box: (x0, y0, x1, y1) XYXY. Internally re-ordered to (y0, x0,
            y1, x1) which is what micro_sam._process_box actually
            consumes.
          image_path, frame_idx: enable caching when both provided.
          multimask_output: when True, SAM evaluates 3 candidates and
            we pick the highest-IoU one.

        Returns:
          (H, W) bool mask.
        """
        from micro_sam.prompt_based_segmentation import segment_from_box as _sfb

        # Resolve embedding (cached or fresh). This also calls self.load()
        # internally on cache miss.
        emb = self.precompute_embedding(
            frame, image_path=image_path, frame_idx=frame_idx)

        frame_rgb = self.to_rgb_uint8(frame)
        x0, y0, x1, y1 = (float(v) for v in box)
        # Re-pack to (y0, x0, y1, x1) for micro_sam.
        box_arr = np.array([y0, x0, y1, x1], dtype=np.float32)
        log('sam_service', 'segment_from_box: starting',
            frame_shape=frame_rgb.shape,
            box_xyxy=[x0, y0, x1, y1],
            box_passed_to_sam_yxyx=box_arr.tolist(),
            multimask=multimask_output,
            cached_emb=(image_path is not None and frame_idx is not None))

        # With image_embeddings passed in, micro_sam's _initialize_predictor
        # calls util.set_precomputed for us — no manual set_image needed.
        try:
            if multimask_output:
                mask, scores, _logits = _sfb(
                    predictor=self._predictor,
                    box=box_arr,
                    image_embeddings=emb,
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
                    image_embeddings=emb,
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
