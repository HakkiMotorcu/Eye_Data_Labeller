"""Tracker subsystem.

A ``TrackerService`` takes the image stack and a per-frame instance
segmentation and returns a remap dict ``{(frame, original_id): track_id}``.
The caller (ToolController) then rewrites Annotation2D ``instance_id``s
and seg.masks pixel labels to make matched cells share an identity
(same name, same color) across frames.

The default tracker is **Trackastra** via micro_sam's
``track_across_frames``. New trackers register themselves in the
``TRACKERS`` dict; the UI builds its combobox and settings panel
from each tracker's ``setting_specs()``.

Adding a custom tracker later:

    class MyTracker(TrackerService):
        name = "MyTracker"
        def setting_specs(self):
            return [SettingSpec('foo', 'Foo', 'int', 5, min=1, max=20)]
        def run(self, timeseries, segmentation, *, on_progress=None):
            ...
            return remap_dict
    TRACKERS[MyTracker.name] = MyTracker
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, List
import numpy as np

from core.debug import log, log_error


@dataclass
class SettingSpec:
    """UI-agnostic description of one tracker setting.

    The View panel renders this into a widget (int -> QSpinBox,
    float -> QDoubleSpinBox, bool -> QCheckBox, choice -> QComboBox).
    Used to keep the tracker code free of any Qt imports.
    """
    key: str
    label: str
    kind: str            # 'int' | 'float' | 'bool' | 'choice'
    default: object
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None
    tooltip: str = ""


class TrackerService(ABC):
    """Base class. Concrete trackers subclass + register in TRACKERS."""

    name: str = "Unnamed"

    def __init__(self):
        self.settings = {spec.key: spec.default for spec in self.setting_specs()}

    @abstractmethod
    def setting_specs(self) -> List[SettingSpec]:
        """Per-tracker settings — drives the UI."""

    @abstractmethod
    def run(self, timeseries: np.ndarray, segmentation: np.ndarray, *,
            on_progress: Optional[Callable] = None) -> dict:
        """Run the tracker.

        Args:
          timeseries:    (T, H, W) uint8 raw image stack.
          segmentation:  (T, H, W) int per-frame instance mask.
          on_progress:   optional callback(n_done, n_total, msg=str).

        Returns:
          remap dict mapping ``(frame_idx, original_id)`` -> ``track_id``.
          ``track_id`` is consistent across frames for the same physical
          cell.
        """

    def update_setting(self, key, value):
        if key in self.settings:
            self.settings[key] = value


class TrackastraTracker(TrackerService):
    """micro_sam.multi_dimensional_segmentation.track_across_frames wrapper.

    This is the default tracker and matches the micro-sam napari plugin
    behavior (Trackastra under the hood). Handles disappearance via
    ``gap_closing``.
    """

    name = "Trackastra (micro-sam default)"

    def setting_specs(self):
        return [
            SettingSpec(
                'mode', 'Mode', 'choice',
                default='greedy', choices=['greedy', 'ilp'],
                tooltip="greedy: per-cell next-edge matching. Fast (seconds).\n"
                        "ilp:    globally optimal association via integer\n"
                        "        linear programming. Slower but better for\n"
                        "        crowded scenes and fast-moving cells.\n"
                        "        Use if greedy is splitting/swapping tracks."),
            SettingSpec(
                'gap_closing', 'Gap closing (frames)', 'int',
                default=3, min=0, max=20, step=1,
                tooltip="Max frames a cell can disappear and still count as "
                        "the same track. 0 = no gap closing (every miss "
                        "starts a new track)."),
            SettingSpec(
                'min_time_extent', 'Min track length (frames)', 'int',
                default=1, min=1, max=100, step=1,
                tooltip="Tracks present in fewer than this many frames are "
                        "discarded after linking."),
        ]

    def run(self, timeseries, segmentation, *, on_progress=None):
        # Bypass micro_sam.track_across_frames (which hardcodes
        # mode='greedy') so we can expose mode as a setting. Use the
        # underlying _tracking_impl directly + _preprocess_closing for
        # gap-closing pre-processing.
        from micro_sam.multi_dimensional_segmentation import (
            _tracking_impl, _preprocess_closing,
        )

        log('tracker.trackastra', 'starting',
            T=int(timeseries.shape[0]),
            HxW=f"{timeseries.shape[-2]}x{timeseries.shape[-1]}",
            n_instances=int(np.unique(segmentation).size - 1),
            settings=dict(self.settings))

        gap = int(self.settings.get('gap_closing', 0))
        mode = str(self.settings.get('mode', 'greedy'))

        seg_for_track = segmentation
        if gap > 0:
            # _preprocess_closing wants a progress-update callable.
            def _noop_pbar(*_a, **_kw):
                pass
            try:
                seg_for_track = _preprocess_closing(
                    segmentation, gap, _noop_pbar)
            except Exception as e:
                log_error('tracker.trackastra',
                          '_preprocess_closing failed', exc=e)
                raise

        # NOTE: micro-sam's _tracking_impl raises NotImplementedError on
        # min_time_extent > 0 (TODO in their code). We always pass None
        # and apply the min-track-length filter ourselves below.
        try:
            tracked, lineage = _tracking_impl(
                timeseries=np.asarray(timeseries),
                segmentation=seg_for_track,
                mode=mode,
                min_time_extent=None,
                output_folder=None,
            )
        except Exception as e:
            log_error('tracker.trackastra', '_tracking_impl raised',
                      exc=e, mode=mode, gap_closing=gap)
            raise

        # Build remap: for each (frame, original_id), look up the track_id
        # at any pixel that had that original_id.
        remap = {}
        T = segmentation.shape[0]
        for t in range(T):
            orig_ids = np.unique(segmentation[t])
            for oid in orig_ids:
                if oid == 0:
                    continue
                ys, xs = np.where(segmentation[t] == oid)
                if len(ys) == 0:
                    continue
                tid = int(tracked[t, ys[0], xs[0]])
                if tid > 0:
                    remap[(int(t), int(oid))] = tid

        # Apply our own min-track-length filter (micro-sam's path is TODO).
        # Drop tracks that span fewer than `min_time_extent` frames — those
        # (frame, oid) pairs simply fall out of the remap, so the caller
        # keeps their original per-frame IDs.
        min_len = int(self.settings.get('min_time_extent', 1))
        if min_len > 1:
            frame_count_by_tid = {}
            for (_, _), tid in remap.items():
                frame_count_by_tid[tid] = frame_count_by_tid.get(tid, 0) + 1
            short_tracks = {tid for tid, c in frame_count_by_tid.items() if c < min_len}
            if short_tracks:
                before = len(remap)
                remap = {k: v for k, v in remap.items() if v not in short_tracks}
                log('tracker.trackastra', 'filtered short tracks',
                    min_len=min_len, dropped_tracks=len(short_tracks),
                    pairs_before=before, pairs_after=len(remap))

        log('tracker.trackastra', 'done',
            n_pairs_mapped=len(remap),
            unique_tracks=len({tid for tid in remap.values()}),
            n_lineage_records=len(lineage) if lineage else 0)
        return remap


# ----- Registry --------------------------------------------------------
TRACKERS = {
    TrackastraTracker.name: TrackastraTracker,
}


def get_default_tracker_name():
    return TrackastraTracker.name


def make_tracker(name):
    cls = TRACKERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tracker: {name!r}. Available: {list(TRACKERS)}")
    return cls()
