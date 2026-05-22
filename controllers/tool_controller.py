import csv
import json
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QColor, QFont, QShortcut, QKeySequence

from core.app_state import AppState

from core.sam_service import (
    SamService, SAM_AVAILABLE, default_sam_hela_path,
    EmbeddingPrecomputeWorker,
)
from core.tracker_service import (
    TRACKERS, make_tracker, get_default_tracker_name, SettingSpec,
)
from core.debug import log, log_error


# ======================================================================
# UNDO / REDO  — lightweight command pattern
# ======================================================================
class UndoStack:
    """Simple undo/redo stack. Max 50 commands."""
    def __init__(self, max_size=50):
        self._undo = []
        self._redo = []
        self._max = max_size

    def push(self, cmd):
        self._undo.append(cmd)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)

    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.redo()
        self._undo.append(cmd)

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def clear(self):
        self._undo.clear()
        self._redo.clear()


# ======================================================================
# UNDO COMMANDS
# ======================================================================
class AddAnnotationCmd:
    def __init__(self, controller, anno):
        self.ctrl = controller
        self.anno = anno

    def undo(self):
        self.ctrl._raw_delete(self.anno)

    def redo(self):
        self.ctrl._raw_restore(self.anno)


class AddAnnotationBatchCmd:
    """Undo/redo adding a group of annotations at once (e.g. vein propagation)."""
    def __init__(self, controller, annos):
        self.ctrl = controller
        self.annos = list(annos)

    def undo(self):
        for anno in self.annos:
            if anno in self.ctrl.annotations:
                anno.delete_ui()
                self.ctrl.annotations.remove(anno)
                if self.ctrl.active_annotation == anno:
                    self.ctrl.active_annotation = None
        self.ctrl._show_frame_annotations(self.ctrl.window._current_frame_idx)

    def redo(self):
        for anno in self.annos:
            if anno not in self.ctrl.annotations:
                self.ctrl.annotations.append(anno)
                self.ctrl.window.view_frame.addItem(anno.roi)
                anno.update_visuals()
        self.ctrl._show_frame_annotations(self.ctrl.window._current_frame_idx)


class DeleteAnnotationCmd:
    def __init__(self, controller, anno, index):
        self.ctrl = controller
        self.anno = anno
        self.index = index

    def undo(self):
        self.ctrl._raw_restore(self.anno, index=self.index)

    def redo(self):
        self.ctrl._raw_delete(self.anno)


class TrackingCmd:
    """Undo/redo a tracker pass.

    Snapshots the seg masks, the instance_colors dict, and each
    annotation's (instance_id, name) before and after the tracker
    applied its remap. Undo restores the 'before' snapshot; redo
    restores 'after'.

    Memory: 2 * seg.masks.nbytes plus small dicts. For typical
    10-frame 500x500 stacks that's ~10 MB — fine.
    """
    def __init__(self, controller, before, after):
        self.ctrl = controller
        self.before = before  # {'masks': np.ndarray, 'colors': dict, 'annos': [(anno, iid, name), ...]}
        self.after = after

    def undo(self):
        self._restore(self.before)

    def redo(self):
        self._restore(self.after)

    def _restore(self, state):
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        seg.masks[:] = state['masks']
        seg.instance_colors.clear()
        seg.instance_colors.update(state['colors'])
        for record in state['annos']:
            # Backwards-compat: old snapshots had 3-tuples without color.
            if len(record) == 4:
                anno, iid, name, color = record
                anno.instance_id = iid
                anno.name = name
                anno.color = color
            else:
                anno, iid, name = record
                anno.instance_id = iid
                anno.name = name
        # Refresh visuals + list
        for anno in self.ctrl.annotations:
            anno.update_visuals()
        self.ctrl._show_frame_annotations(self.ctrl.window._current_frame_idx)
        self.ctrl.window._update_seg_overlay()
        self.ctrl.state.annotations_changed.emit()


class SamBoxPromptCmd:
    """Undo/redo for a single SAM-box-prompt paint into one cell.

    Snapshots the affected frame's full mask AND the annotation's ROI
    geometry (because the bbox may have been tightened to fit the new
    mask). Memory cost is one (H, W) int32 per command — fine.
    """
    def __init__(self, controller, frame_idx, instance_id,
                 before_frame, after_frame,
                 before_geom=None, after_geom=None, class_type='cell'):
        self.ctrl = controller
        self.frame_idx = int(frame_idx)
        self.instance_id = int(instance_id)
        self.before_frame = before_frame  # (H, W) int32 copy
        self.after_frame = after_frame
        self.before_geom = before_geom    # {'x','y','w','h'} or None
        self.after_geom = after_geom
        self.class_type = class_type

    def _restore(self, frame_mask, geom):
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        seg.get_layer(self.class_type)[self.frame_idx][:] = frame_mask
        for a in self.ctrl.annotations:
            if (a.frame_idx == self.frame_idx
                    and a.instance_id == self.instance_id
                    and a.class_type == self.class_type):
                if geom is not None:
                    a._is_syncing = True
                    a.roi.setPos([geom['x'], geom['y']])
                    a.roi.setSize([geom['w'], geom['h']])
                    a._is_syncing = False
                a.update_visuals()
                break
        if self.frame_idx == self.ctrl.window._current_frame_idx:
            self.ctrl.window._update_seg_overlay()
        self.ctrl.state.annotations_changed.emit()

    def undo(self):
        self._restore(self.before_frame, self.before_geom)

    def redo(self):
        self._restore(self.after_frame, self.after_geom)


class DeletePaintOnlyIdentityCmd:
    """Undo/redo deletion of every frame-entry + every painted pixel for a
    paint-only identity (vessel / capillary).

    Snapshots the (T, H, W) boolean mask of pixels that belonged to this
    instance_id so undo can restore them exactly.
    """
    def __init__(self, controller, annos, instance_id, pixel_mask, color,
                 class_type='vessel'):
        self.ctrl = controller
        self.annos = list(annos)
        self.iid = instance_id
        self.pixel_mask = pixel_mask  # (T, H, W) bool, in `class_type` layer
        self.color = color
        self.class_type = class_type

    def undo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            layer = seg.get_layer(self.class_type)
            layer[self.pixel_mask] = self.iid
            if self.color is not None:
                seg.register_instance_color(self.iid, self.color,
                                             class_type=self.class_type)
        for anno in self.annos:
            self.ctrl._raw_restore(anno)
        self.ctrl._show_frame_annotations(self.ctrl.window._current_frame_idx)
        self.ctrl.window._update_seg_overlay()
        self.ctrl.state.annotations_changed.emit()

    def redo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            layer = seg.get_layer(self.class_type)
            layer[self.pixel_mask] = 0
            seg.get_colors(self.class_type).pop(self.iid, None)
        if self.ctrl.active_annotation in self.annos:
            self.ctrl.active_annotation = None
        for anno in self.annos:
            self.ctrl._raw_delete(anno)
        self.ctrl._show_frame_annotations(self.ctrl.window._current_frame_idx)
        self.ctrl.window._update_seg_overlay()
        self.ctrl.state.annotations_changed.emit()


class MoveResizeCmd:
    """Undo/redo a bbox move or resize, optionally also restoring pixel data.

    old_mask / new_mask are full frame mask copies (int32 ndarray).
    Provided only when pixels were actually modified (translate or crop).
    """
    def __init__(self, anno, old_state, new_state,
                 ctrl=None, frame_idx=None, old_mask=None, new_mask=None,
                 class_type='cell'):
        self.anno = anno
        self.old = old_state
        self.new = new_state
        self.ctrl = ctrl
        self.frame_idx = frame_idx
        self.old_mask = old_mask
        self.new_mask = new_mask
        self.class_type = class_type

    def undo(self):
        self._apply_box(self.old)
        self._apply_mask(self.old_mask)

    def redo(self):
        self._apply_box(self.new)
        self._apply_mask(self.new_mask)

    def _apply_box(self, s):
        a = self.anno
        a._is_syncing = True
        a.roi.setPos([s['x'], s['y']])
        a.roi.setSize([s['w'], s['h']])
        a._is_syncing = False
        a.sig_updated.emit(a)

    def _apply_mask(self, mask):
        if mask is None or self.ctrl is None:
            return
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        seg.get_layer(self.class_type)[self.frame_idx] = mask.copy()
        self.ctrl.window._update_seg_overlay()


class LockCmd:
    def __init__(self, controller, anno, locked_after):
        self.ctrl = controller
        self.anno = anno
        self.locked_after = locked_after

    def undo(self):
        self.anno.set_locked(not self.locked_after)
        self.ctrl._refresh_list_colors()

    def redo(self):
        self.anno.set_locked(self.locked_after)
        self.ctrl._refresh_list_colors()


class BrushStrokeCmd:
    """Undo/redo a single brush stroke (press → drag → release)."""
    def __init__(self, controller, frame_idx, old_mask, new_mask,
                 class_type='cell'):
        self.ctrl = controller
        self.frame_idx = frame_idx
        self.old_mask = old_mask
        self.new_mask = new_mask
        self.class_type = class_type

    def undo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            seg.get_layer(self.class_type)[self.frame_idx] = self.old_mask.copy()
            self.ctrl.window._update_seg_overlay()

    def redo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            seg.get_layer(self.class_type)[self.frame_idx] = self.new_mask.copy()
            self.ctrl.window._update_seg_overlay()


class PropagateWithSpawnCmd:
    """Undo/redo a propagate operation that also created the per-frame
    paint-only annotations (when the user invoked Propagate Mask on a
    vessel/capillary that only existed on the current frame).

    Single undoable step:
      undo  → revert propagated pixels, then remove the spawned annos.
      redo  → re-add the spawned annos, then re-apply pixels.
    """
    def __init__(self, controller, annos, old_masks, new_masks, class_type):
        self.ctrl = controller
        self.annos = list(annos)
        self.old_masks = old_masks
        self.new_masks = new_masks
        self.class_type = class_type

    def _apply_masks(self, masks_dict):
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        layer = seg.get_layer(self.class_type)
        for fi, mask in masks_dict.items():
            layer[fi] = mask.copy()

    def undo(self):
        self._apply_masks(self.old_masks)
        for anno in self.annos:
            if anno in self.ctrl.annotations:
                anno.delete_ui()
                self.ctrl.annotations.remove(anno)
                if self.ctrl.active_annotation == anno:
                    self.ctrl.active_annotation = None
        self.ctrl._show_frame_annotations(
            self.ctrl.window._current_frame_idx)
        self.ctrl.window._update_seg_overlay()

    def redo(self):
        for anno in self.annos:
            if anno not in self.ctrl.annotations:
                self.ctrl.annotations.append(anno)
                self.ctrl.window.view_frame.addItem(anno.roi)
                anno.update_visuals()
        self._apply_masks(self.new_masks)
        self.ctrl._show_frame_annotations(
            self.ctrl.window._current_frame_idx)
        self.ctrl.window._update_seg_overlay()


class PropagateMaskCmd:
    """Undo/redo a mask-propagation operation across multiple frames."""
    def __init__(self, controller, old_masks, new_masks, class_type='cell'):
        # old_masks / new_masks: dict {frame_idx: np.ndarray copy}
        self.ctrl = controller
        self.old_masks = old_masks   # {fi: mask before propagation}
        self.new_masks = new_masks   # {fi: mask after  propagation}
        self.class_type = class_type

    def _apply(self, masks_dict):
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        layer = seg.get_layer(self.class_type)
        for fi, mask in masks_dict.items():
            layer[fi] = mask.copy()
        self.ctrl.window._update_seg_overlay()

    def undo(self):
        self._apply(self.old_masks)

    def redo(self):
        self._apply(self.new_masks)


# ======================================================================
# 2D ANNOTATION
# ======================================================================
class Annotation2D(QObject):
    sig_clicked = pyqtSignal(object)
    sig_updated = pyqtSignal(object)

    # class_type slot values:
    #   'cell'       - has a bbox, drawn via ROI
    #   'vessel'     - paint-only (no bbox); larger retinal vessel
    #   'capillary'  - paint-only (no bbox); small retinal capillary
    # Non-cell classes use the same paint-only rendering path.

    def __init__(self, name, view, controller,
                 start_pos=(100, 100), start_size=(40, 40),
                 shape_mode='rect', frame_idx=0,
                 instance_id=None, color=None, class_type='cell'):
        super().__init__()
        self.name = name
        self.view = view
        self.controller = controller
        self.shape_mode = shape_mode
        self.frame_idx = frame_idx
        self.instance_id = instance_id  # seg instance ID (if from seg map)
        self.color = color              # (R, G, B) display colour
        # Backwards compat: 'vein' was the old name for 'vessel'
        if class_type == 'vein':
            class_type = 'vessel'
        self.class_type = class_type

        self._seg_dirty = False         # True when seg pixels were painted/erased
        self._is_syncing = False
        self.is_selected = False
        self.is_locked = False
        self.is_hidden = False  # per-row visibility toggle (D)

        x, y = start_pos
        w, h = start_size

        self.roi = self._make_roi(shape_mode, [x, y], [w, h])
        self.view.addItem(self.roi)
        self._connect_roi_signals()

        if self.is_paint_only:
            # Paint-only classes have no visible bbox
            self.roi.setPen(pg.mkPen(None))
            self.roi.hoverPen = pg.mkPen(None)
            self.roi.translatable = False
            self.roi.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for h in self.roi.getHandles():
                h.setVisible(False)
            self.roi.setVisible(False)
        else:
            self.update_visuals()

    @property
    def is_paint_only(self):
        """True for non-cell classes (vessel, capillary): paint-only, no visible bbox."""
        return self.class_type != 'cell'

    @staticmethod
    def _make_rect_roi(pos, size):
        roi = pg.RectROI(pos, size, resizable=False)
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi.addScaleHandle([1, 1], [0, 0])
        return roi

    @staticmethod
    def _make_ellipse_roi(pos, size):
        roi = pg.EllipseROI(pos, size)
        old_children = list(roi.childItems())
        for h in roi.getHandles()[:]:
            roi.removeHandle(h)
        for c in roi.childItems():
            if c in old_children:
                c.setParentItem(None)
        roi.addScaleHandle([0.5, 0],   [0.5, 1])
        roi.addScaleHandle([1,   0.5], [0,   0.5])
        roi.addScaleHandle([0.5, 1],   [0.5, 0])
        roi.addScaleHandle([0,   0.5], [1,   0.5])
        return roi

    @staticmethod
    def _make_roi(mode, pos, size):
        if mode == 'ellipse':
            return Annotation2D._make_ellipse_roi(pos, size)
        return Annotation2D._make_rect_roi(pos, size)

    def _connect_roi_signals(self):
        self.roi.sigRegionChanged.connect(self._on_region_changed)
        self.roi.sigClicked.connect(self._on_interaction)
        self.roi.sigRegionChangeStarted.connect(self._on_interaction)
        self.roi.sigRegionChangeStarted.connect(self._on_drag_start)
        self.roi.sigRegionChangeFinished.connect(self._on_drag_end)

    def _disconnect_roi_signals(self):
        try:
            self.roi.sigRegionChanged.disconnect(self._on_region_changed)
            self.roi.sigClicked.disconnect(self._on_interaction)
            self.roi.sigRegionChangeStarted.disconnect(self._on_interaction)
            self.roi.sigRegionChangeStarted.disconnect(self._on_drag_start)
            self.roi.sigRegionChangeFinished.disconnect(self._on_drag_end)
        except RuntimeError:
            pass

    def set_shape_mode(self, mode):
        if mode == self.shape_mode:
            return
        x, y = self.roi.pos()
        w, h = self.roi.size()

        self._disconnect_roi_signals()
        self.view.removeItem(self.roi)

        self.shape_mode = mode
        self.roi = self._make_roi(mode, [x, y], [w, h])
        self.view.addItem(self.roi)
        self._connect_roi_signals()

        if self.is_locked:
            self.roi.translatable = False
            self.roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.update_visuals()

    def set_locked(self, locked):
        self.is_locked = locked
        self.roi.translatable = not locked
        # Reject mouse buttons entirely when locked so neither the ROI body
        # nor a stray hit on a handle moves anything. pyqtgraph handles still
        # receive mouse events when only hidden, so also disable them.
        self.roi.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton if locked else Qt.MouseButton.LeftButton)
        for h in self.roi.getHandles():
            h.setEnabled(not locked)
        self.update_visuals()

    def update_visuals(self):
        if self.is_paint_only:
            # Veins never show ROI
            return
        style = Qt.PenStyle.SolidLine

        if self.controller._label_color_mode:
            # Simple status-based coloring: R=unlocked, Y=selected, G=locked
            if self.is_locked:
                color = QColor(0, 200, 0)          # green
                width = 3 if self.is_selected else 2
                if self.is_selected:
                    style = Qt.PenStyle.DashLine
            elif self.is_selected:
                color = QColor(255, 255, 0)         # yellow
                width = 2
            else:
                color = QColor(255, 50, 50)         # red
                width = 2
            z_val = 10 if self.is_selected else (5 if self.is_locked else 2)
        elif self.is_locked:
            color = 'g'
            width = 3 if self.is_selected else 1
            if self.is_selected:
                style = Qt.PenStyle.DashLine
            z_val = 5
        elif self.is_selected:
            color = 'y'
            width = 2
            z_val = 10
        elif self.color:
            color = QColor(*self.color)
            width = 1
            z_val = 2
        else:
            color = 'r'
            width = 1
            z_val = 2

        if color is not None:
            pen = pg.mkPen(color, width=width, style=style)
            hover = pg.mkPen('y', width=width + 1)
        else:
            pen = pg.mkPen(None)
            hover = pg.mkPen(None)

        self.roi.setPen(pen)
        self.roi.hoverPen = hover
        self.roi.setZValue(z_val)
        for h in self.roi.getHandles():
            h.setVisible(not self.is_locked)

    def _on_interaction(self, *args):
        if not self.is_selected:
            self.sig_clicked.emit(self)

    def _on_region_changed(self):
        if not self._is_syncing:
            self.sig_updated.emit(self)

    def _on_drag_start(self, *args):
        self.controller._geometry_snapshot = ToolController._snap_geometry(self)

    def _on_drag_end(self, *args):
        if self.is_locked:
            # Belt-and-braces: set_locked already blocks mouse input, but
            # if anything sneaks through we refuse to push a MoveResizeCmd
            # for a locked annotation.
            self.controller._geometry_snapshot = None
            return
        old = self.controller._geometry_snapshot
        if old is None:
            return
        new = ToolController._snap_geometry(self)
        self.controller._geometry_snapshot = None

        # Nothing actually changed
        if (old['x'] == new['x'] and old['y'] == new['y'] and
                old['w'] == new['w'] and old['h'] == new['h']):
            return

        ctrl = self.controller
        seg = ctrl.window.seg_data
        frame = self.frame_idx
        has_seg = (seg is not None and self.instance_id is not None
                   and not self.is_paint_only)

        # Detect translate vs resize (0.5 px tolerance on size)
        _EPS = 0.5
        size_changed = (abs(old['w'] - new['w']) > _EPS or
                        abs(old['h'] - new['h']) > _EPS)

        old_mask = new_mask = None

        ct = self.class_type
        layer = seg.get_layer(ct) if seg is not None else None
        if not size_changed:
            # ── TRANSLATE: move pixels with the box ───────────────────────
            if has_seg:
                scale = ctrl._seg_scale()
                if scale is not None:
                    sx, sy = scale
                    old_mask = layer[frame].copy()
                    seg.move_instance_pixels(
                        frame, self.instance_id,
                        (old['x'] * sx, old['y'] * sy,
                         old['w'] * sx, old['h'] * sy),
                        (new['x'] * sx, new['y'] * sy,
                         new['w'] * sx, new['h'] * sy),
                        class_type=ct,
                    )
                    new_mask = layer[frame].copy()
                    ctrl.window._update_seg_overlay()
        else:
            # ── RESIZE ────────────────────────────────────────────────────
            shrunk = (new['w'] < old['w'] - _EPS or new['h'] < old['h'] - _EPS)
            if shrunk and has_seg:
                # Smaller box → crop pixels that now fall outside
                scale = ctrl._seg_scale()
                if scale is not None:
                    sx, sy = scale
                    old_mask = layer[frame].copy()
                    seg.crop_instance_to_bbox(
                        frame, self.instance_id,
                        new['x'] * sx, new['y'] * sy,
                        new['w'] * sx, new['h'] * sy,
                        class_type=ct,
                    )
                    new_mask = layer[frame].copy()
                    ctrl.window._update_seg_overlay()
            # Bigger box → pixels unchanged, no mask snapshot needed

        ctrl._undo_stack.push(MoveResizeCmd(
            self, old, new,
            ctrl=ctrl if old_mask is not None else None,
            frame_idx=frame,
            old_mask=old_mask,
            new_mask=new_mask,
            class_type=ct,
        ))

    def delete_ui(self):
        self.view.removeItem(self.roi)

    def set_visible(self, visible):
        if self.is_paint_only:
            return  # vein ROIs always hidden
        self.roi.setVisible(visible)


# ======================================================================
# TOOL CONTROLLER
# ======================================================================
class ToolController:
    SIZE_PRESETS = {
        1: (10, 10),
        2: (20, 20),
        3: (40, 40),
        4: (80, 80),
    }

    def __init__(self, main_window):
        self.window = main_window
        self.state = AppState()
        self.annotations = []
        self.active_annotation = None
        self.anno_counter = 0
        self.current_shape_mode = 'rect'
        self._last_size = (40, 40)
        self._undo_stack = UndoStack()
        self._geometry_snapshot = None
        self._label_color_mode = False  # status-based R/Y/G coloring
        self._seg_edit_mode = 'select'  # 'select' | 'paint' | 'erase'
        self._is_painting = False       # True while mouse is pressed in paint/erase mode
        self._brush_cursor = None       # circle item showing brush on view
        self._brush_mask_snapshot = None # mask copy before current brush stroke
        self._brush_snapshot_class = None # class layer that snapshot came from
        # Auto-save dirty tracking for smart-mode mask flushing.
        self._seg_dirty_since_save = False
        import time as _time_mod
        self._last_mask_save_ts = _time_mod.monotonic()
        self._last_save_at = None  # wall-clock ts of last explicit save
        # Periodic refresh of the I/O panel save-status label so the
        # "Saved 2 min ago" text rolls forward without user action.
        self._save_status_timer = QTimer()
        self._save_status_timer.timeout.connect(self._refresh_save_status)
        self._save_status_timer.start(10_000)
        # SAM service — lazy-loaded on first use. Default model is the
        # collaborators' fine-tuned ViT-B (sam_hela). User can swap via
        # the SAM section in the View panel.
        self.sam_service = SamService(
            model_type='vit_b', checkpoint_path=default_sam_hela_path())
        # Tracker service — default Trackastra via micro-sam.
        self.tracker_service = make_tracker(get_default_tracker_name())
        # Embedding precompute worker (only one alive at a time — SAM
        # predictor isn't thread-safe).
        self._embed_worker = None
        # Frame queued by _on_frame_changed_embed while a worker is busy.
        # Picked up in _on_embed_finished_ok so we never block the UI
        # thread waiting for an in-flight compute.
        self._embed_pending_frame = None
        # Application-modal "SAM model running" dialog. Created on demand
        # by _show_embed_dialog and torn down on finish / error / cancel.
        self._embed_dialog = None
        # Annotation list filter: set of class keys currently shown.
        # Defaults to "everything"; users can toggle classes independently
        # from the filter button row (B). "All" is a master shortcut.
        self._list_filter = {'cell', 'vessel', 'capillary'}

        # --- Button connections ---
        self.window.btn_add.clicked.connect(self.spawn_new_annotation)
        self.window.btn_add_vessel.clicked.connect(self.spawn_vessel)
        self.window.btn_add_capillary.clicked.connect(self.spawn_capillary)
        self.window.btn_delete.clicked.connect(self.delete_selected)
        self.window.btn_rename.clicked.connect(self._start_rename)
        self.window.btn_fit_bbox.clicked.connect(self._manual_fit_bbox)
        self.window.btn_fix_names.clicked.connect(self._fix_names_clicked)
        self.window.list_annotations.itemChanged.connect(self._on_list_item_edited)
        self.window.list_annotations.itemClicked.connect(self._on_list_item_clicked)
        # Class filter buttons (B): All / Cells / Vessels / Capillaries.
        # Map button → filter key; the handler reads from this dict so we
        # don't rely on Qt dynamic properties (which were not propagating
        # in PyQt6 on macOS).
        self._filter_btn_map = {
            self.window.btn_filter_all:       'all',
            self.window.btn_filter_cell:      'cell',
            self.window.btn_filter_vessel:    'vessel',
            self.window.btn_filter_capillary: 'capillary',
        }
        for btn, key in self._filter_btn_map.items():
            # ToolController is not a QObject, so self.sender() isn't
            # available — bind the key explicitly via a default arg.
            btn.clicked.connect(
                lambda _checked=False, k=key: self._on_filter_button_clicked(k))

        # Seg-overlay z-order buttons (MC6): pick which class renders on top.
        self._zorder_btn_map = {
            self.window.btn_zorder_cell:      'cell',
            self.window.btn_zorder_vessel:    'vessel',
            self.window.btn_zorder_capillary: 'capillary',
        }
        for btn, key in self._zorder_btn_map.items():
            btn.clicked.connect(
                lambda _checked=False, k=key: self._on_zorder_button_clicked(k))
        # Apply the initial visual state (All → every button lit).
        self._apply_filter_visual()
        self.window.btn_lock.clicked.connect(self.lock_active)
        self.window.btn_unlock.clicked.connect(self.unlock_active)
        self.window.btn_lock_all.clicked.connect(self.lock_all)
        self.window.btn_unlock_all.clicked.connect(self.unlock_all)
        self.window.btn_hide_locked.clicked.connect(self.toggle_hide_locked)
        self.window.btn_label_colors.clicked.connect(self.toggle_label_colors)
        self.window.btn_export_cells.clicked.connect(self.export_cells)
        self.window.btn_export_veins.clicked.connect(self.export_veins)
        self.window.btn_export_all.clicked.connect(self.export_all)
        self.window.btn_export_coco.clicked.connect(self.export_coco_sidecar)
        self.window.btn_import.clicked.connect(self.load_annotations)
        # Load actions: Project (folder picker) vs Single Class (file picker).
        self.window.btn_load_project.clicked.connect(self.load_project_folder)
        self.window.btn_load_class.clicked.connect(self.load_single_class_tif)
        self.window.btn_io_settings.clicked.connect(self.open_io_settings)
        self.window.btn_run_sam.clicked.connect(self.run_sam_segmentation)
        self.window.combo_sam_model.currentIndexChanged.connect(self._on_sam_model_changed)
        self.window.btn_sam_precompute.clicked.connect(self.precompute_all_frames)
        self.window.btn_sam_precompute.setEnabled(SamService.available())
        # Auto-precompute current frame whenever the timeline moves.
        self.state.frame_changed.connect(self._on_frame_changed_embed)
        self._refresh_sam_status()

        # SAM auto-link enable mirrors the All-frames toggle (single-
        # frame SAM has no temporal info to track over).
        self.window.chk_sam_all_frames.toggled.connect(
            self.window.chk_sam_auto_link.setEnabled)

        # Tracking section wiring.
        for tname in TRACKERS:
            self.window.combo_tracker.addItem(tname)
        self.window.combo_tracker.setCurrentText(self.tracker_service.name)
        self.window.combo_tracker.currentTextChanged.connect(self._on_tracker_changed)
        self.window.btn_run_tracker.clicked.connect(self.run_tracking_now)
        self._rebuild_tracker_settings()
        self.window.slider_seg_opacity.valueChanged.connect(self._on_seg_opacity_changed)
        # Mirror seg-opacity between the View-panel slider and the Tools
        # mirror. Each updates the other with signals blocked to avoid
        # ping-pong.
        self.window.slider_seg_opacity_tools.valueChanged.connect(
            self._on_seg_opacity_tools_changed)
        self.window.btn_toggle_seg.clicked.connect(self._on_toggle_seg)
        self.window.btn_toggle_seg_tools.clicked.connect(self._on_toggle_seg)
        # Seg editing connections
        self.window._seg_mode_group.idClicked.connect(self._on_seg_mode_changed)
        self.window.slider_brush_size.valueChanged.connect(self._on_brush_size_changed)
        self.window.btn_fill_bbox.clicked.connect(self.fill_bbox_cmd)
        self.window.btn_sam_box.clicked.connect(self.run_sam_box_prompt)
        self.window.btn_clear_seg_mask.clicked.connect(self.clear_seg_mask_for_selected)
        self.window.btn_save_seg.clicked.connect(self.save_seg_map)
        self.window.btn_propagate_mask.clicked.connect(self.propagate_vein_mask)

        # Display controls
        self.window.slider_min.valueChanged.connect(self._on_level_slider)
        self.window.slider_max.valueChanged.connect(self._on_level_slider)
        self.window.btn_auto_levels.clicked.connect(self._auto_levels)
        self.window.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)

        # Timeline controls
        self.window.slider_timeline.valueChanged.connect(self._on_timeline_changed)
        self.window.spin_frame.valueChanged.connect(self._on_spin_frame_changed)
        self.window.btn_frame_first.clicked.connect(self._go_first_frame)
        self.window.btn_frame_prev.clicked.connect(self._go_prev_frame)
        self.window.btn_frame_next.clicked.connect(self._go_next_frame)
        self.window.btn_frame_last.clicked.connect(self._go_last_frame)

        # Annotation list selection
        self.window.list_annotations.currentItemChanged.connect(self._on_list_item_changed)

        # --- Keyboard shortcuts ---
        self._shortcuts = [
            QShortcut(QKeySequence("A"),          self.window, self.spawn_new_annotation),
            QShortcut(QKeySequence("L"),          self.window, self.lock_active),
            QShortcut(QKeySequence("U"),          self.window, self.unlock_active),
            QShortcut(QKeySequence("H"),          self.window, self._shortcut_hide_locked),
            QShortcut(QKeySequence("R"),          self.window, self.reset_zoom),
            QShortcut(QKeySequence("Delete"),     self.window, self.delete_selected),
            QShortcut(QKeySequence("Backspace"),  self.window, self.delete_selected),
            QShortcut(QKeySequence("Ctrl+S"),     self.window, self.save_annotations),
            QShortcut(QKeySequence("Ctrl+I"),     self.window, self.load_annotations),
            QShortcut(QKeySequence("N"),          self.window, self.select_next_annotation),
            QShortcut(QKeySequence("P"),          self.window, self.select_prev_annotation),
            QShortcut(QKeySequence("1"),          self.window, lambda: self.apply_size_preset(1)),
            QShortcut(QKeySequence("2"),          self.window, lambda: self.apply_size_preset(2)),
            QShortcut(QKeySequence("3"),          self.window, lambda: self.apply_size_preset(3)),
            QShortcut(QKeySequence("4"),          self.window, lambda: self.apply_size_preset(4)),
            QShortcut(QKeySequence("T"),          self.window, self.apply_last_size),
            QShortcut(QKeySequence("0"),          self.window, self.capture_current_size),
            QShortcut(QKeySequence("Ctrl+L"),     self.window, self.lock_and_advance),
            QShortcut(QKeySequence("Ctrl+Z"),     self.window, self.undo),
            QShortcut(QKeySequence("Ctrl+Shift+Z"), self.window, self.redo),
            # Frame navigation
            QShortcut(QKeySequence("Right"),      self.window, self._go_next_frame),
            QShortcut(QKeySequence("Left"),       self.window, self._go_prev_frame),
            QShortcut(QKeySequence("Home"),       self.window, self._go_first_frame),
            QShortcut(QKeySequence("End"),        self.window, self._go_last_frame),
            # Seg editing modes
            QShortcut(QKeySequence("D"),          self.window, lambda: self._set_seg_mode('paint')),
            QShortcut(QKeySequence("E"),          self.window, lambda: self._set_seg_mode('erase')),
            QShortcut(QKeySequence("Escape"),     self.window, lambda: self._set_seg_mode('select')),
            QShortcut(QKeySequence("F"),          self.window, self.fill_bbox_cmd),
            QShortcut(QKeySequence("B"),          self.window, self.run_sam_box_prompt),
            QShortcut(QKeySequence("Shift+E"),    self.window, self.clear_seg_mask_for_selected),
            QShortcut(QKeySequence("V"),          self.window, self.spawn_vessel),
            QShortcut(QKeySequence("C"),          self.window, self.spawn_capillary),
            QShortcut(QKeySequence("X"),          self.window, self._shortcut_toggle_force_paint),
            QShortcut(QKeySequence("Ctrl+P"),     self.window, self.propagate_vein_mask),
        ]

        # Double-click to place annotation
        self._connect_view_double_click()

        # Mouse events for seg brush painting
        self._connect_brush_events()

        # Auto-save timer (every 60 seconds)
        self._autosave_path = None
        self._autosave_timer = QTimer()
        self._autosave_timer.timeout.connect(self._autosave)
        self._apply_autosave_interval()  # reads QSettings, defaults to 30s

    # ------------------------------------------------------------------
    # FRAME NAVIGATION
    # ------------------------------------------------------------------
    def _on_timeline_changed(self, value):
        self._maybe_fit_bbox(self.active_annotation)
        self.window.spin_frame.blockSignals(True)
        self.window.spin_frame.setValue(value)
        self.window.spin_frame.blockSignals(False)
        self.window.display_frame(value)
        self._show_frame_annotations(value)
        self.state.frame_changed.emit(value)

    def _on_spin_frame_changed(self, value):
        self._maybe_fit_bbox(self.active_annotation)
        self.window.slider_timeline.blockSignals(True)
        self.window.slider_timeline.setValue(value)
        self.window.slider_timeline.blockSignals(False)
        self.window.display_frame(value)
        self._show_frame_annotations(value)
        self.state.frame_changed.emit(value)

    def _go_first_frame(self):
        self.window.slider_timeline.setValue(0)

    def _go_last_frame(self):
        if self.window.video_data:
            self.window.slider_timeline.setValue(self.window.video_data.num_frames - 1)

    def _go_prev_frame(self):
        v = self.window.slider_timeline.value()
        if v > 0:
            self.window.slider_timeline.setValue(v - 1)

    def _go_next_frame(self):
        v = self.window.slider_timeline.value()
        if self.window.video_data and v < self.window.video_data.num_frames - 1:
            self.window.slider_timeline.setValue(v + 1)

    # ------------------------------------------------------------------
    # PER-FRAME ANNOTATION VISIBILITY
    # ------------------------------------------------------------------
    def _show_frame_annotations(self, frame_idx):
        """Show only annotations for the given frame, hide the rest.
        Rebuild the list widget to match."""
        from PyQt6.QtWidgets import QTreeWidgetItem
        from ui.main_window import make_swatch_icon

        self.window.list_annotations.blockSignals(True)
        self.window.list_annotations.clear()

        visible_annos = []
        class_labels = {'cell': 'Cell', 'vessel': 'Vessel',
                        'capillary': 'Capillary'}
        flt = self._list_filter  # set of class keys currently shown
        seg = self.window.seg_data
        for anno in self.annotations:
            if anno.frame_idx == frame_idx:
                # Hidden / class-filtered annotations still get their ROI
                # state updated so they vanish from the viewer.
                hidden = getattr(anno, 'is_hidden', False)
                if anno.is_paint_only or hidden:
                    anno.roi.setVisible(False)
                else:
                    anno.roi.setVisible(True)
                # Skip rows whose class isn't in the active filter set.
                if anno.class_type not in flt:
                    continue
                # For paint-only classes (vessel / capillary) the seg
                # pixels ARE the annotation. An entry with no pixels on
                # this frame is normally a phantom (e.g. from spawning
                # with "all frames" and only painting one) and gets
                # hidden — BUT a brand-new annotation that has no pixels
                # anywhere yet is the user's freshly-spawned vessel
                # they're about to paint into, so it stays visible.
                if (anno.is_paint_only and seg is not None
                        and anno.instance_id is not None):
                    layer = seg.get_layer(anno.class_type)
                    here = np.any(layer[frame_idx] == anno.instance_id)
                    if not here:
                        anywhere = np.any(layer == anno.instance_id)
                        if anywhere:
                            continue  # phantom on this frame
                visible_annos.append(anno)
                cls = class_labels.get(anno.class_type, anno.class_type)
                vis_glyph  = " " if hidden else "●"
                lock_glyph = "🔒" if anno.is_locked else ""
                item = QTreeWidgetItem(
                    ["", anno.name, cls, vis_glyph, lock_glyph])
                item.setIcon(0, make_swatch_icon(anno.color))
                # Only the name column should be editable (not the
                # swatch / glyph columns).
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(3, Qt.AlignmentFlag.AlignCenter)
                item.setTextAlignment(4, Qt.AlignmentFlag.AlignCenter)
                self.window.list_annotations.addTopLevelItem(item)
            else:
                anno.roi.setVisible(False)

        # Re-select the active annotation if it's on this frame
        if self.active_annotation and self.active_annotation.frame_idx == frame_idx:
            items = self.window.list_annotations.findItems(
                self.active_annotation.name, Qt.MatchFlag.MatchExactly, 1)
            if items:
                self.window.list_annotations.setCurrentItem(items[0])
        else:
            # Try to follow the same instance to this frame so a tracked
            # cell stays selected as the user scrubs. Fall back to the
            # first visible annotation when no match exists.
            prev = self.active_annotation
            follow = None
            if prev is not None:
                if prev.instance_id is not None:
                    # (class_type, instance_id) is the real identity key —
                    # iid alone collides across classes after the per-class
                    # namespace fix.
                    follow = next(
                        (a for a in visible_annos
                         if a.instance_id == prev.instance_id
                         and a.class_type == prev.class_type),
                        None)
                if follow is None:
                    # Paint-only annos may share a name across frames.
                    follow = next(
                        (a for a in visible_annos
                         if a.name == prev.name
                         and a.class_type == prev.class_type),
                        None)
            self.active_annotation = None
            target = follow or (visible_annos[0] if visible_annos else None)
            if target is not None:
                self.active_annotation = target
                self.active_annotation.is_selected = True
                self.active_annotation.update_visuals()
                items = self.window.list_annotations.findItems(
                    target.name, Qt.MatchFlag.MatchExactly, 1)
                if items:
                    self.window.list_annotations.setCurrentItem(items[0])

        self.window.list_annotations.blockSignals(False)
        self._refresh_list_colors()
        self._update_stats()
        self.update_inspector()
        # Always keep the seg overlay in sync with the current annotation state
        self.window._update_seg_overlay()

    def _get_frame_annotations(self):
        """Return annotations for the current frame only."""
        cur = self.window._current_frame_idx
        return [a for a in self.annotations if a.frame_idx == cur]

    # ------------------------------------------------------------------
    # SEGMENTATION MASK HELPERS
    # ------------------------------------------------------------------
    def _seg_scale(self):
        """Return (sx, sy) from video coords → seg coords, or None."""
        seg = self.window.seg_data
        vid = self.window.video_data
        if seg is None or vid is None:
            return None
        sx = seg.width / vid.width
        sy = seg.height / vid.height
        return sx, sy

    def _fit_bbox_to_seg(self, anno):
        """Resize the annotation's ROI to tightly fit its seg pixels."""
        if anno is None or anno.instance_id is None:
            return
        if anno.is_paint_only:
            return  # veins have no bbox to fit
        seg = self.window.seg_data
        if seg is None:
            return
        bbox = seg.get_instance_bbox(anno.frame_idx, anno.instance_id,
                                       class_type=anno.class_type)
        if bbox is None:
            return  # no pixels — keep current bbox
        sx0, sy0, sw, sh = bbox
        # Convert seg coords → video coords
        scale = self._seg_scale()
        if scale is None:
            return
        sx, sy = scale
        vx = sx0 / sx
        vy = sy0 / sy
        vw = sw / sx
        vh = sh / sy
        anno._is_syncing = True
        anno.roi.setPos([vx, vy])
        anno.roi.setSize([vw, vh])
        anno._is_syncing = False
        anno.sig_updated.emit(anno)

    def _maybe_fit_bbox(self, anno):
        """Fit bbox if the annotation's seg pixels were modified."""
        if anno is not None and anno._seg_dirty:
            self._fit_bbox_to_seg(anno)
            anno._seg_dirty = False

    def _erase_seg_for_anno(self, anno):
        """Zero-out seg mask pixels for an annotation's instance & bbox."""
        scale = self._seg_scale()
        if scale is None or anno.instance_id is None:
            return
        seg = self.window.seg_data
        sx, sy = scale
        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        # Convert video coords → seg coords
        seg.erase_bbox(anno.frame_idx, anno.instance_id,
                       x * sx, y * sy, w * sx, h * sy,
                       class_type=anno.class_type)

    def _move_seg_pixels(self, anno, old_geom, new_geom):
        """Move seg mask pixels when a bbox is dragged or resized."""
        if anno.is_paint_only:
            return  # veins don't move with bbox
        scale = self._seg_scale()
        if scale is None or anno.instance_id is None:
            return
        seg = self.window.seg_data
        sx, sy = scale
        old_bbox = (old_geom['x'] * sx, old_geom['y'] * sy,
                    old_geom['w'] * sx, old_geom['h'] * sy)
        new_bbox = (new_geom['x'] * sx, new_geom['y'] * sy,
                    new_geom['w'] * sx, new_geom['h'] * sy)

        # Pure translation vs resize
        moved = (old_geom['x'] != new_geom['x'] or old_geom['y'] != new_geom['y'])
        resized = (old_geom['w'] != new_geom['w'] or old_geom['h'] != new_geom['h'])

        if resized:
            seg.resize_instance_pixels(anno.frame_idx, anno.instance_id,
                                       old_bbox, new_bbox,
                                       class_type=anno.class_type)
        elif moved:
            seg.move_instance_pixels(anno.frame_idx, anno.instance_id,
                                     old_bbox, new_bbox,
                                     class_type=anno.class_type)
        self.window._update_seg_overlay()

    # ------------------------------------------------------------------
    # ANNOTATION CRUD HELPERS
    # ------------------------------------------------------------------
    def open_io_settings(self):
        """Open the modal output / autosave settings dialog. On accept,
        re-apply the autosave timer interval so changes take effect
        without a restart."""
        from ui.io_settings_dialog import IOSettingsDialog
        dlg = IOSettingsDialog(self.window)
        if dlg.exec():
            self._apply_autosave_interval()

    # ----- Project I/O helpers ------------------------------------------
    # Dirty-mask tracking — set True every time seg pixels change, reset
    # whenever the project is saved (explicit Save or smart autosave).
    # Defined as an instance attribute in __init__; helpers below.
    def _mark_seg_dirty(self):
        self._seg_dirty_since_save = True
        self._refresh_save_status()

    def _mark_seg_clean(self):
        self._seg_dirty_since_save = False
        import time
        self._last_mask_save_ts = time.monotonic()
        self._last_save_at = time.time()
        self._refresh_save_status()

    def _refresh_save_status(self):
        """Update the I/O panel's save status label."""
        lbl = getattr(self.window, 'lbl_save_status', None)
        if lbl is None:
            return
        import time
        if self._seg_dirty_since_save:
            lbl.setText("● Unsaved changes")
            lbl.setStyleSheet(
                "color: #e9a33a; font-family: monospace; font-size: 11px;")
            return
        last = getattr(self, '_last_save_at', None)
        if last is None:
            lbl.setText("Not saved yet")
            lbl.setStyleSheet(
                "color: #888; font-family: monospace; font-size: 11px;")
            return
        secs = int(time.time() - last)
        if secs < 60:
            ago = f"{secs}s ago"
        elif secs < 3600:
            ago = f"{secs // 60} min ago"
        else:
            ago = f"{secs // 3600} h ago"
        lbl.setText(f"✓ Saved {ago}")
        lbl.setStyleSheet(
            "color: #2a9d8f; font-family: monospace; font-size: 11px;")

    def _io_settings(self):
        """Return a ``(mode, custom_root)`` tuple from QSettings."""
        from PyQt6.QtCore import QSettings
        from core import project_io
        s = QSettings()
        mode = s.value(project_io.SETTING_OUTPUT_MODE,
                        project_io.DEFAULTS[project_io.SETTING_OUTPUT_MODE])
        root = s.value(project_io.SETTING_OUTPUT_CUSTOM_ROOT,
                        project_io.DEFAULTS[project_io.SETTING_OUTPUT_CUSTOM_ROOT])
        return str(mode or project_io.OUTPUT_MODE_SUBFOLDER), str(root or "")

    def _resolve_out_folder(self, video_path=None):
        """Return the per-video output folder for the active session.

        Used by every save / load / autosave / resume path so the
        location is consistent.
        """
        from core import project_io
        video_path = video_path or self.window._current_file
        if not video_path:
            return ""
        mode, root = self._io_settings()
        return project_io.resolve_output_folder(video_path, mode, root)

    def _class_counts_for_manifest(self):
        """Per-class *unique-instance* counts for the project manifest."""
        seen = {'cell': set(), 'vessel': set(), 'capillary': set()}
        for a in self.annotations:
            if a.instance_id is None:
                continue
            ct = getattr(a, 'class_type', 'cell')
            if ct in seen:
                seen[ct].add(int(a.instance_id))
        return {k: len(v) for k, v in seen.items()}

    def _ensure_seg_data(self):
        """Guarantee ``window.seg_data`` exists once a video is loaded.

        Paint-only classes (vessel, capillary) need a segmentation layer to
        write into. We previously created the seg layer only when loading a
        pre-existing mask file or running SAM — opening a raw video and
        clicking Vessel/Capillary did nothing because there was no seg to
        paint. Now we auto-create an empty layer on first annotation.
        """
        if self.window.seg_data is not None:
            return self.window.seg_data
        vd = self.window.video_data
        if vd is None:
            return None
        from core.volume_data import SegmentationData
        self.window.seg_data = SegmentationData.empty(vd.width, vd.height, vd.num_frames)
        self.window._seg_visible = True
        return self.window.seg_data

    # Class → name prefix used by the gap-fill namer and the normalizer.
    _CLASS_PREFIX = {'cell': 'Cell', 'vessel': 'Vessel', 'capillary': 'Capillary'}

    def _fix_names_clicked(self):
        """User clicked Fix names — normalize then re-render the list."""
        renamed, recolored = self._normalize_anno_names_and_colors()
        self._show_frame_annotations(self.window._current_frame_idx)
        self.window._update_seg_overlay()
        QMessageBox.information(
            self.window, "Fix Names",
            f"Renamed {renamed} annotation(s); recolored {recolored}."
            if (renamed or recolored)
            else "Every annotation already matches its class — no changes.")

    def _normalize_anno_names_and_colors(self):
        """Force every annotation's name and color into agreement with
        its class_type.

        Repairs stale state from before the per-class instance-ID
        namespace fix: an annotation whose name starts with the wrong
        class prefix is renamed to ``{Class}_{iid}``, and its color is
        re-registered (or pulled) from the matching per-class palette.

        Safe to call repeatedly — it's idempotent on a well-formed
        session.
        """
        seg = self.window.seg_data
        renamed = recolored = 0
        for anno in self.annotations:
            ct = getattr(anno, 'class_type', 'cell')
            prefix = self._CLASS_PREFIX.get(ct, 'Cell')
            iid = getattr(anno, 'instance_id', None)

            # --- Name ---
            current = anno.name or ''
            head = current.split('_', 1)[0]
            if head not in self._CLASS_PREFIX.values() or head != prefix:
                # Wrong prefix (or none) → coerce to {Class}_{iid}, falling
                # back to a gap-filled number when iid is None.
                if iid is not None:
                    new_name = f"{prefix}_{int(iid)}"
                else:
                    _, new_name = self._next_available_name(prefix)
                if new_name != current:
                    anno.name = new_name
                    renamed += 1

            # --- Color ---
            if seg is not None and iid is not None:
                colors = seg.get_colors(ct)
                fresh = seg.register_instance_color(int(iid), class_type=ct)
                if anno.color != fresh:
                    anno.color = fresh
                    recolored += 1
                # Make sure no leftover color is registered for this iid in
                # the *other* classes' palettes (cleanup from the old
                # shared-namespace state).
                for other in self._CLASS_PREFIX:
                    if other == ct:
                        continue
                    other_colors = seg.get_colors(other)
                    if int(iid) in other_colors:
                        # Only pop when the *other* class has no annotation
                        # owning this iid — otherwise we'd recolor a real
                        # annotation by surprise.
                        owned = any(
                            a.class_type == other and a.instance_id == iid
                            for a in self.annotations
                        )
                        if not owned:
                            other_colors.pop(int(iid), None)

        if renamed or recolored:
            log('controller.normalize',
                'fixed stale state',
                renamed=renamed, recolored=recolored)
        return renamed, recolored

    def _next_available_name(self, prefix):
        """Lowest unused integer suffix for ``{prefix}_{n}``.

        Naming is gap-filling: with Cell_1, Cell_4 present, the next add is
        Cell_2; after that Cell_3, then Cell_5. Each prefix (Cell / Vessel /
        Capillary) has its own number space.

        Annotations on multiple frames sharing one identity (same
        instance_id) are counted once.
        """
        used = set()
        needle = f"{prefix}_"
        for anno in self.annotations:
            if not anno.name.startswith(needle):
                continue
            tail = anno.name[len(needle):]
            try:
                used.add(int(tail))
            except ValueError:
                # SAM may produce 'Cell_5_2' — count the primary number only.
                head = tail.split('_', 1)[0]
                try:
                    used.add(int(head))
                except ValueError:
                    pass
        n = 1
        while n in used:
            n += 1
        return n, f"{prefix}_{n}"

    # ------------------------------------------------------------------
    # ANNOTATION CRUD
    # ------------------------------------------------------------------
    def spawn_new_annotation(self, start_pos=None):
        if isinstance(start_pos, bool):
            start_pos = None
        if not self.window.video_data:
            return

        frame_idx = self.window._current_frame_idx

        # Auto-create the seg layer if this is the first annotation on a raw
        # video. Needed so cells get an instance_id assigned (and so
        # vessels/capillaries can be painted at all).
        self._ensure_seg_data()

        if start_pos is None:
            start_pos = self._next_default_spawn_pos(frame_idx)

        # --- Tracking: try to match a cell from the previous frame ---
        match = self._find_tracking_match(frame_idx, start_pos)

        if match is not None:
            name = match['name']
            instance_id = match['instance_id']
            color = match['color']
            # Make sure seg color is registered
            seg = self.window.seg_data
            if seg is not None and instance_id is not None:
                seg.register_instance_color(instance_id, color)
        else:
            # No match — assign a new instance with a gap-filled name.
            seg = self.window.seg_data
            instance_id = None
            color = None
            if seg is not None:
                instance_id = seg.next_instance_id()
                color = seg.register_instance_color(instance_id)
                self.anno_counter = max(self.anno_counter, instance_id)
            n, name = self._next_available_name('Cell')
            self.anno_counter = max(self.anno_counter, n)

        new_anno = Annotation2D(
            name, self.window.view_frame, self,
            start_pos=start_pos,
            start_size=self._last_size,
            shape_mode=self.current_shape_mode,
            frame_idx=frame_idx,
            instance_id=instance_id,
            color=color,
        )

        new_anno.sig_clicked.connect(self.select_annotation)
        new_anno.sig_updated.connect(self._on_anno_updated)
        self.annotations.append(new_anno)
        self._show_frame_annotations(frame_idx)
        self.select_annotation(new_anno)
        self._undo_stack.push(AddAnnotationCmd(self, new_anno))
        self.state.annotations_changed.emit()

    # Per-class palettes for paint-only retinal structures.
    # Indexed by gap-fill number so the same Vessel_N always gets the same
    # shade across reopens of the same file.
    _VESSEL_PALETTE = [
        (120, 80, 200), (145, 95, 215), (105, 65, 180), (135, 105, 225),
        (115, 75, 195), (155, 115, 230), (100, 85, 210), (125, 90, 200),
        (140, 80, 190), (110, 60, 175), (150, 105, 220), (130, 95, 205),
    ]
    _CAPILLARY_PALETTE = [
        (235, 130, 200), (245, 155, 215), (220, 110, 185), (250, 170, 225),
        (215, 105, 175), (255, 180, 230), (228, 140, 195), (240, 150, 210),
        (210, 120, 180), (245, 165, 220), (225, 135, 195), (250, 175, 225),
    ]

    def _shade_for(self, class_type, index):
        if class_type == 'vessel':
            return self._VESSEL_PALETTE[(index - 1) % len(self._VESSEL_PALETTE)]
        if class_type == 'capillary':
            return self._CAPILLARY_PALETTE[(index - 1) % len(self._CAPILLARY_PALETTE)]
        return (180, 180, 180)

    def spawn_vessel(self):
        """Create a paint-only vessel annotation, optionally propagated across frames."""
        self._spawn_paint_only('vessel', 'Vessel')

    def spawn_capillary(self):
        """Create a paint-only capillary annotation, optionally propagated across frames."""
        self._spawn_paint_only('capillary', 'Capillary')

    # Backwards-compat alias for any external callers still using the old name.
    def spawn_vein(self):
        self.spawn_vessel()

    def _spawn_paint_only(self, class_type, name_prefix):
        """Shared spawn path for non-cell, paint-only retinal structures."""
        if not self.window.video_data:
            return
        frame_idx = self.window._current_frame_idx
        total_frames = self.window.video_data.num_frames

        # Paint-only classes have no bbox; the seg layer IS their only
        # representation. Auto-create one if the user opened a raw video.
        seg = self._ensure_seg_data()

        # Pick the gap-fill name first, then choose a shade keyed by N so
        # the same Vessel_N color is reproducible across sessions.
        n, name = self._next_available_name(name_prefix)
        self.anno_counter = max(self.anno_counter, n)
        default_color = self._shade_for(class_type, n)

        instance_id = None
        color = None
        if seg is not None:
            # Pull the instance_id from the class's own layer namespace and
            # register the color in that class's color table — otherwise
            # vessel/capillary IDs collide with cells and the overlay
            # falls back to the generic palette (looks wrong).
            instance_id = seg.next_instance_id(class_type=class_type)
            color = seg.register_instance_color(
                instance_id, color=default_color, class_type=class_type)
            self.anno_counter = max(self.anno_counter, instance_id)

        # --- Propagation dialog ---
        propagate_all = False
        if total_frames > 1:
            reply = QMessageBox.question(
                self.window,
                f"Propagate {name_prefix}?",
                f"Add '{name}' to all {total_frames} frames?\n\n"
                "Recommended workflow:\n"
                "  1. Click Yes so the annotation exists on every frame.\n"
                f"  2. Paint the {class_type} mask on the current frame.\n"
                "  3. Press Ctrl+P (or 'Propagate Mask') to copy those pixels\n"
                "     to all other frames at once.\n"
                "  4. Navigate frame-by-frame to fine-tune where needed.\n\n"
                "• Yes — create annotation on all frames (pixels still empty until you paint + propagate).\n"
                "• No  — annotation only on the current frame.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            propagate_all = (reply == QMessageBox.StandardButton.Yes)

        frame_range = range(total_frames) if propagate_all else range(frame_idx, frame_idx + 1)

        created = []
        for fi in frame_range:
            anno = Annotation2D(
                name, self.window.view_frame, self,
                start_pos=(0, 0),
                start_size=(1, 1),
                shape_mode='rect',
                frame_idx=fi,
                instance_id=instance_id,
                color=color,
                class_type=class_type,
            )
            anno.sig_clicked.connect(self.select_annotation)
            anno.sig_updated.connect(self._on_anno_updated)
            self.annotations.append(anno)
            anno.roi.setVisible(False)
            created.append(anno)

        self._undo_stack.push(AddAnnotationBatchCmd(self, created))
        self._show_frame_annotations(frame_idx)

        current = next(
            (a for a in created if a.frame_idx == frame_idx), None)
        if current:
            self.select_annotation(current)

        self._set_seg_mode('paint')
        self.state.annotations_changed.emit()

    def _get_visible_center(self):
        vb = self.window.view_frame.getView()
        r = vb.viewRange()
        cx = (r[0][0] + r[0][1]) / 2.0
        cy = (r[1][0] + r[1][1]) / 2.0
        if self.window.video_data:
            cx = max(0, min(cx, self.window.video_data.width - 1))
            cy = max(0, min(cy, self.window.video_data.height - 1))
        return cx, cy

    _SPAWN_STEP = 18.0   # px offset between consecutive default-position spawns

    def _next_default_spawn_pos(self, frame_idx):
        """Pick a non-stacking default position for a new bbox on this frame.

        Without staggering, repeated A presses all land at the visible
        center -> bboxes overlap exactly and only the topmost is
        clickable. Offset by ``_SPAWN_STEP`` from the most recent cell
        on this frame; wrap into the visible viewport if it goes off
        the right/bottom edge.
        """
        cx, cy = self._get_visible_center()
        w, h = self._last_size
        base = (cx - w / 2.0, cy - h / 2.0)

        # Find the most recent cell on this frame to offset from.
        siblings = [a for a in self.annotations
                    if a.frame_idx == frame_idx and not a.is_paint_only]
        if not siblings:
            return base

        last = siblings[-1]
        lx, ly = last.roi.pos()
        px = float(lx) + self._SPAWN_STEP
        py = float(ly) + self._SPAWN_STEP

        vb = self.window.view_frame.getView()
        (x_lo, x_hi), (y_lo, y_hi) = vb.viewRange()
        if px + w > x_hi or py + h > y_hi:
            # Wrap back into the viewport.
            px, py = x_lo + self._SPAWN_STEP, y_lo + self._SPAWN_STEP

        if self.window.video_data:
            W = self.window.video_data.width
            H = self.window.video_data.height
            px = max(0, min(px, W - w))
            py = max(0, min(py, H - h))
        return (px, py)

    # ------------------------------------------------------------------
    # TRACKING — match new annotation to previous frame
    # ------------------------------------------------------------------
    _TRACK_MAX_DIST = 60  # max centroid distance (video px) to consider a match

    def _find_tracking_match(self, frame_idx, start_pos):
        """Look for a cell in the previous frame that is close to *start_pos*.

        Uses two strategies:
        1. Seg-overlap: if a seg map is loaded, check which instance's pixels
           overlap the new bbox region in the *previous* frame.
        2. Centroid distance: find the nearest annotation centroid.

        Returns dict {name, instance_id, color} or None if no match.
        """
        if frame_idx <= 0:
            return None

        prev_frame = frame_idx - 1
        prev_annos = [a for a in self.annotations
                      if a.frame_idx == prev_frame and a.class_type == 'cell']
        if not prev_annos:
            return None

        sx, sy = start_pos
        w, h = self._last_size
        new_cx = sx + w / 2
        new_cy = sy + h / 2

        seg = self.window.seg_data
        scale = self._seg_scale()

        # --- Strategy 1: seg overlap ---
        best_overlap_anno = None
        if seg is not None and scale is not None:
            scx, scy = scale
            # Check which instance ID is most frequent under the new bbox in prev frame
            prev_mask = seg.get_mask(prev_frame)
            # New bbox in seg coords
            r0 = max(0, int(sy * scy))
            r1 = min(seg.height, int((sy + h) * scy))
            c0 = max(0, int(sx * scx))
            c1 = min(seg.width, int((sx + w) * scx))
            if r1 > r0 and c1 > c0:
                region = prev_mask[r0:r1, c0:c1]
                ids_in_region = region[region != 0]
                if len(ids_in_region) > 0:
                    vals, counts = np.unique(ids_in_region, return_counts=True)
                    dominant_id = int(vals[np.argmax(counts)])
                    # Find annotation with that instance_id in prev frame
                    for a in prev_annos:
                        if a.instance_id == dominant_id:
                            best_overlap_anno = a
                            break

        # --- Strategy 2: nearest centroid ---
        best_dist = float('inf')
        best_centroid_anno = None
        for a in prev_annos:
            ax, ay = a.roi.pos()
            aw, ah = a.roi.size()
            acx = ax + aw / 2
            acy = ay + ah / 2
            d = ((acx - new_cx) ** 2 + (acy - new_cy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_centroid_anno = a

        # Prefer seg overlap match, fall back to centroid
        match_anno = best_overlap_anno or best_centroid_anno
        if match_anno is None:
            return None

        # Check distance threshold (even for seg-overlap, the matched anno
        # should be reasonably close)
        ax, ay = match_anno.roi.pos()
        aw, ah = match_anno.roi.size()
        dist = (((ax + aw / 2) - new_cx) ** 2 +
                ((ay + ah / 2) - new_cy) ** 2) ** 0.5

        # For seg-overlap matches allow larger distance
        max_d = self._TRACK_MAX_DIST * (2 if best_overlap_anno else 1)
        if dist > max_d:
            return None

        # Check this name isn't already used on the current frame
        existing_names = {a.name for a in self.annotations
                         if a.frame_idx == frame_idx}
        if match_anno.name in existing_names:
            return None

        return {
            'name': match_anno.name,
            'instance_id': match_anno.instance_id,
            'color': match_anno.color,
        }

    def reset_zoom(self):
        self.window.view_frame.getView().autoRange()

    # ------------------------------------------------------------------
    # RESIZE / SIZE PRESETS
    # ------------------------------------------------------------------
    def apply_size_preset(self, preset_key):
        anno = self.active_annotation
        if anno is None or anno.is_locked:
            return
        old_snap = self._snap_geometry(anno)
        pw, ph = self.SIZE_PRESETS.get(preset_key, (40, 40))
        self._last_size = (pw, ph)
        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        cx = x + w / 2
        cy = y + h / 2
        anno._is_syncing = True
        anno.roi.setPos([cx - pw / 2, cy - ph / 2])
        anno.roi.setSize([pw, ph])
        anno._is_syncing = False
        anno.sig_updated.emit(anno)
        self._undo_stack.push(MoveResizeCmd(anno, old_snap, self._snap_geometry(anno)))

    def apply_last_size(self):
        anno = self.active_annotation
        if anno is None or anno.is_locked:
            return
        old_snap = self._snap_geometry(anno)
        pw, ph = self._last_size
        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        cx = x + w / 2
        cy = y + h / 2
        anno._is_syncing = True
        anno.roi.setPos([cx - pw / 2, cy - ph / 2])
        anno.roi.setSize([pw, ph])
        anno._is_syncing = False
        anno.sig_updated.emit(anno)
        self._undo_stack.push(MoveResizeCmd(anno, old_snap, self._snap_geometry(anno)))

    def capture_current_size(self):
        anno = self.active_annotation
        if anno is None:
            return
        w, h = anno.roi.size()
        self._last_size = (round(w, 1), round(h, 1))
        print(f"Captured size: W={self._last_size[0]}, H={self._last_size[1]}")

    # ------------------------------------------------------------------
    # UNDO / REDO
    # ------------------------------------------------------------------
    @staticmethod
    def _snap_geometry(anno):
        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        return {'x': x, 'y': y, 'w': w, 'h': h}

    def _raw_delete(self, anno):
        if anno not in self.annotations:
            return
        anno.delete_ui()
        self.annotations.remove(anno)
        if self.active_annotation == anno:
            self.active_annotation = None
        self._show_frame_annotations(self.window._current_frame_idx)

    def _raw_restore(self, anno, index=None):
        anno.view = self.window.view_frame
        self.window.view_frame.addItem(anno.roi)
        if index is not None and 0 <= index <= len(self.annotations):
            self.annotations.insert(index, anno)
        else:
            self.annotations.append(anno)
        anno.update_visuals()
        self._show_frame_annotations(self.window._current_frame_idx)
        if anno.frame_idx == self.window._current_frame_idx:
            self.select_annotation(anno)

    def undo(self):
        self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()

    def select_next_annotation(self):
        frame_annos = self._get_frame_annotations()
        if not frame_annos:
            return
        if self.active_annotation is None or self.active_annotation not in frame_annos:
            idx = 0
        else:
            idx = (frame_annos.index(self.active_annotation) + 1) % len(frame_annos)
        self.select_annotation(frame_annos[idx])

    def select_prev_annotation(self):
        frame_annos = self._get_frame_annotations()
        if not frame_annos:
            return
        if self.active_annotation is None or self.active_annotation not in frame_annos:
            idx = len(frame_annos) - 1
        else:
            idx = (frame_annos.index(self.active_annotation) - 1) % len(frame_annos)
        self.select_annotation(frame_annos[idx])

    def lock_and_advance(self):
        if not self.active_annotation:
            return
        self.lock_active()
        frame_annos = self._get_frame_annotations()
        if not frame_annos:
            return
        start = frame_annos.index(self.active_annotation) if self.active_annotation in frame_annos else 0
        for i in range(1, len(frame_annos) + 1):
            candidate = frame_annos[(start + i) % len(frame_annos)]
            if not candidate.is_locked:
                self.select_annotation(candidate)
                return

    # ------------------------------------------------------------------
    # SELECTION / LOCKING
    # ------------------------------------------------------------------
    def select_annotation(self, annotation):
        if self.active_annotation == annotation:
            return
        # Auto-fit bbox of the previously selected annotation if seg-dirty
        self._maybe_fit_bbox(self.active_annotation)
        self.active_annotation = annotation
        cur = self.window._current_frame_idx
        for anno in self.annotations:
            if anno.frame_idx == cur:
                anno.is_selected = (anno == annotation)
                anno.update_visuals()
        items = self.window.list_annotations.findItems(annotation.name, Qt.MatchFlag.MatchExactly, 1)
        if items:
            self.window.list_annotations.blockSignals(True)
            self.window.list_annotations.setCurrentItem(items[0])
            self.window.list_annotations.blockSignals(False)
        self.update_inspector()
        self._refresh_list_colors()
        if self._label_color_mode:
            self.window._update_seg_overlay()

    def _on_list_item_changed(self, current, previous):
        if current:
            name = current.text(1)
            cur = self.window._current_frame_idx
            for anno in self.annotations:
                if anno.name == name and anno.frame_idx == cur:
                    self.select_annotation(anno)
                    break

    def _on_list_item_clicked(self, item, column):
        """Per-row eye/lock glyph clicks toggle anno state (D)."""
        if column not in (3, 4):
            return
        name = item.text(1)
        cur = self.window._current_frame_idx
        anno = next((a for a in self.annotations
                     if a.name == name and a.frame_idx == cur), None)
        if anno is None:
            return
        if column == 3:
            anno.is_hidden = not getattr(anno, 'is_hidden', False)
            # Bbox ROI follows hidden state immediately; seg overlay
            # rebuild also needs to drop hidden instances.
            anno.roi.setVisible(
                (not anno.is_paint_only) and (not anno.is_hidden))
            self.window._update_seg_overlay()
        else:  # column == 4 → lock toggle
            anno.set_locked(not anno.is_locked)
            anno.update_visuals()
        # Repaint the row glyphs + recompute stats.
        self._show_frame_annotations(cur)

    _ALL_CLASS_KEYS = {'cell', 'vessel', 'capillary'}

    def _on_filter_button_clicked(self, key):
        """Class-filter button (B) clicked; rebuild list.

        Behavior:
          - "All": master shortcut. If everything is currently in the
            filter set, clear it. Otherwise restore the full set.
          - "Cell/Vessel/Capillary": independent toggle on that class.
        """
        if key not in ('all',) and key not in self._ALL_CLASS_KEYS:
            return
        log('controller.list_filter',
            'filter clicked', key=key,
            before=sorted(self._list_filter))
        if key == 'all':
            if self._list_filter == self._ALL_CLASS_KEYS:
                self._list_filter = set()
            else:
                self._list_filter = set(self._ALL_CLASS_KEYS)
        else:
            new = set(self._list_filter)
            if key in new:
                new.remove(key)
            else:
                new.add(key)
            self._list_filter = new
        log('controller.list_filter',
            'filter updated', after=sorted(self._list_filter))
        self._apply_filter_visual()
        self._show_frame_annotations(self.window._current_frame_idx)

    # ------------------------------------------------------------------
    # Seg overlay z-order (MC6)
    # ------------------------------------------------------------------
    # Default depth order (bottom to top, before any "lift to top" pick).
    # Vessels are the largest/deepest structure, capillaries above,
    # cells shallowest — so cell-on-top is the natural default.
    _NATURAL_DEPTH = ('vessel', 'capillary', 'cell')

    def _on_zorder_button_clicked(self, top):
        """User picked a layer to render on top. The other two retain
        their natural depth ordering beneath it."""
        if top not in self._NATURAL_DEPTH:
            return
        bottom = [ct for ct in self._NATURAL_DEPTH if ct != top]
        new_order = (*bottom, top)
        if new_order == getattr(self.window, '_seg_layer_order', None):
            # No change — still re-sync the visuals so an idle click on
            # the already-selected button keeps it lit.
            self._apply_zorder_visual()
            return
        self.window._seg_layer_order = new_order
        log('controller.zorder', 'layer order changed',
            order=list(new_order))
        self._apply_zorder_visual()
        self.window._update_seg_overlay()

    def _apply_zorder_visual(self):
        """Light up the button matching the current top layer; others off."""
        order = getattr(self.window, '_seg_layer_order',
                         self._NATURAL_DEPTH)
        top = order[-1]
        for btn, key in self._zorder_btn_map.items():
            btn.blockSignals(True)
            btn.setChecked(key == top)
            btn.blockSignals(False)

    def _apply_filter_visual(self):
        """Sync filter-button checked states to the underlying set.

        - "All" is lit when every class is in the filter.
        - Each class button is lit when its class is in the filter.
        """
        flt = self._list_filter
        for btn, key in self._filter_btn_map.items():
            if key == 'all':
                checked = (flt == self._ALL_CLASS_KEYS)
            else:
                checked = (key in flt)
            btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(False)

    def _on_list_item_edited(self, item, column=1):
        """Called when a list item is renamed via inline editing."""
        # Only the Name column (1) carries rename data.
        if column != 1:
            return
        new_name = item.text(1).strip()
        if not new_name:
            # Revert to old name
            if self.active_annotation:
                item.setText(1, self.active_annotation.name)
            return
        anno = self.active_annotation
        if anno is None:
            return
        if new_name == anno.name:
            return
        anno.name = new_name
        self._show_frame_annotations(self.window._current_frame_idx)

    def _start_rename(self):
        """Begin inline editing of the selected list item."""
        item = self.window.list_annotations.currentItem()
        if item:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.window.list_annotations.editItem(item, 1)

    def _manual_fit_bbox(self):
        """Manually fit the active annotation's bbox to its seg pixels."""
        anno = self.active_annotation
        if anno is None:
            return
        self._fit_bbox_to_seg(anno)
        anno._seg_dirty = False

    def _on_anno_updated(self, annotation):
        self.update_inspector(annotation)
        self._update_stats()

    def update_inspector(self, annotation=None):
        if not self.active_annotation:
            self.window.lbl_coords.setText("No annotation selected")
            return
        anno = self.active_annotation
        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        cls = anno.class_type.capitalize()
        iid = anno.instance_id or '—'
        if anno.is_paint_only:
            self.window.lbl_coords.setText(
                f"{cls}  ID: {iid}  Locked: {anno.is_locked}\n"
                f"(paint-only — no bbox)")
        else:
            self.window.lbl_coords.setText(
                f"Pos: ({int(x)}, {int(y)})  Size: ({int(w)}, {int(h)})\n"
                f"{cls}  ID: {iid}  {anno.shape_mode}  Locked: {anno.is_locked}")

    def lock_active(self):
        if self.active_annotation and not self.active_annotation.is_locked:
            # Auto-fit bbox to actual seg pixels before locking
            self._maybe_fit_bbox(self.active_annotation)
            self._undo_stack.push(LockCmd(self, self.active_annotation, True))
            self.active_annotation.set_locked(True)
            self._refresh_list_colors()
            self._update_stats()
            if self._label_color_mode:
                self.window._update_seg_overlay()

    def unlock_active(self):
        if self.active_annotation and self.active_annotation.is_locked:
            self._undo_stack.push(LockCmd(self, self.active_annotation, False))
            self.active_annotation.set_locked(False)
            self.active_annotation.set_visible(True)
            self._refresh_list_colors()
            self._update_stats()
            if self._label_color_mode:
                self.window._update_seg_overlay()

    def lock_all(self):
        for anno in self._get_frame_annotations():
            anno.set_locked(True)
        self._refresh_list_colors()
        self._update_stats()

    def unlock_all(self):
        for anno in self._get_frame_annotations():
            anno.set_locked(False)
            anno.set_visible(True)
        self._refresh_list_colors()
        self._update_stats()

    def delete_selected(self):
        if self.active_annotation not in self.annotations:
            return
        target = self.active_annotation

        # Paint-only annotations (vessel / capillary) are multi-frame:
        # one Annotation2D per frame, all sharing instance_id. Deleting
        # one frame's entry while the other 11 stay creates ghost masks
        # and breaks the gap-fill namer. Delete the whole identity.
        if target.is_paint_only and target.instance_id is not None:
            self._delete_paint_only_identity(target.instance_id,
                                              target.class_type)
        else:
            index = self.annotations.index(target)
            self._erase_seg_for_anno(target)
            target.delete_ui()
            self.annotations.remove(target)
            self.active_annotation = None
            self._undo_stack.push(DeleteAnnotationCmd(self, target, index))

        self._show_frame_annotations(self.window._current_frame_idx)
        self.window._update_seg_overlay()
        self.state.annotations_changed.emit()

    def _delete_paint_only_identity(self, instance_id, class_type):
        """Remove every frame-entry and every painted pixel for the given
        (class_type, instance_id) identity. Records an undoable command
        that can restore both the annotations and the pixels.

        class_type is required because instance IDs live in per-class
        namespaces — a vessel iid=4 and a capillary iid=4 are different
        annotations and must not be co-deleted.
        """
        seg = self.window.seg_data
        pixel_mask = None
        color = None

        victims = [a for a in self.annotations
                   if a.instance_id == instance_id
                   and a.class_type == class_type]
        if seg is not None:
            layer = seg.get_layer(class_type)
            colors = seg.get_colors(class_type)
            pixel_mask = (layer == instance_id)
            color = colors.get(int(instance_id))
            layer[pixel_mask] = 0
            colors.pop(int(instance_id), None)

        if self.active_annotation in victims:
            self.active_annotation = None
        for anno in victims:
            anno.delete_ui()
            if anno in self.annotations:
                self.annotations.remove(anno)

        if pixel_mask is not None and victims:
            self._undo_stack.push(
                DeletePaintOnlyIdentityCmd(self, victims, int(instance_id),
                                            pixel_mask, color,
                                            class_type=class_type))

    def toggle_hide_locked(self, checked):
        for anno in self._get_frame_annotations():
            if anno.is_locked:
                anno.set_visible(not checked)

    def toggle_label_colors(self, checked):
        self._label_color_mode = checked
        for anno in self._get_frame_annotations():
            anno.update_visuals()
        self._refresh_list_colors()
        self.window._update_seg_overlay()

    def _shortcut_hide_locked(self):
        btn = self.window.btn_hide_locked
        btn.setChecked(not btn.isChecked())
        self.toggle_hide_locked(btn.isChecked())

    def _shortcut_toggle_force_paint(self):
        btn = self.window.btn_force_paint
        btn.setChecked(not btn.isChecked())

    # ------------------------------------------------------------------
    # DOUBLE-CLICK TO PLACE  /  SINGLE-CLICK SEG PIXEL SELECTION
    # ------------------------------------------------------------------
    def _connect_view_double_click(self):
        vb = self.window.view_frame.getView()
        vb.scene().sigMouseClicked.connect(self._on_view_clicked)

    def _on_view_clicked(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._seg_edit_mode != 'select':
            return

        vb = self.window.view_frame.getView()
        pos = vb.mapSceneToView(ev.scenePos())

        if ev.double():
            # Double-click → place new annotation
            ev.accept()
            w, h = self._last_size
            start_pos = (pos.x() - w / 2, pos.y() - h / 2)
            self.spawn_new_annotation(start_pos=start_pos)
        else:
            # Single click that wasn't already claimed by a ROI → try
            # to select the annotation whose seg mask pixel was clicked.
            if not ev.isAccepted():
                self._try_select_by_seg_pixel(pos.x(), pos.y())

    def _try_select_by_seg_pixel(self, vx, vy):
        """Select the annotation whose instance-ID occupies pixel (vx, vy)
        in the segmentation mask.  Works for veins and cells alike."""
        seg = self.window.seg_data
        if seg is None:
            return

        frame = self.window._current_frame_idx
        scale = self._seg_scale()
        if scale is not None:
            sx, sy = scale
            cx, cy = int(round(vx * sx)), int(round(vy * sy))
        else:
            cx, cy = int(round(vx)), int(round(vy))

        if cx < 0 or cy < 0 or cx >= seg.width or cy >= seg.height:
            return

        # Walk class layers top-down (cell wins over capillary over vessel
        # by default) so the click goes to whatever the user actually sees
        # on top in the overlay.
        order = getattr(self.window, '_seg_layer_order',
                        ('vessel', 'capillary', 'cell'))
        for ct in reversed(order):  # top to bottom
            layer = seg.get_layer(ct)
            if layer is None:
                continue
            iid = int(layer[frame, cy, cx])
            if iid == 0:
                continue
            for anno in self.annotations:
                if (anno.frame_idx == frame
                        and anno.instance_id == iid
                        and anno.class_type == ct):
                    self.select_annotation(anno)
                    return
            # No annotation owns these pixels — keep looking down.

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------
    def _update_stats(self):
        total = len(self.annotations)
        if total == 0:
            self.window.lbl_stats.setText("No annotations")
            return
        cur = self.window._current_frame_idx
        seg = self.window.seg_data
        frame_annos = [a for a in self.annotations if a.frame_idx == cur]
        # For paint-only classes count only entries that actually have
        # pixels on this frame — a phantom entry without painted pixels
        # is invisible and shouldn't inflate the panel count.
        def _has_pixels(a):
            if not a.is_paint_only:
                return True
            if seg is None or a.instance_id is None:
                return False
            return bool(np.any(
                seg.get_layer(a.class_type)[cur] == a.instance_id))
        n_cell  = sum(1 for a in frame_annos if a.class_type == 'cell')
        n_vess  = sum(1 for a in frame_annos
                       if a.class_type == 'vessel' and _has_pixels(a))
        n_cap   = sum(1 for a in frame_annos
                       if a.class_type == 'capillary' and _has_pixels(a))
        # Hide zero-count classes so the line stays terse.
        parts = []
        if n_cell:
            parts.append(f"{n_cell} cell" + ("s" if n_cell != 1 else ""))
        if n_vess:
            parts.append(f"{n_vess} vessel" + ("s" if n_vess != 1 else ""))
        if n_cap:
            parts.append(f"{n_cap} capillar" + ("ies" if n_cap != 1 else "y"))
        head = " · ".join(parts) if parts else "0 on frame"
        locked = sum(1 for a in frame_annos if a.is_locked)
        self.window.lbl_stats.setText(
            f"{head}   |   Locked: {locked}   |   All frames: {total}")

    def _refresh_list_colors(self):
        from ui.main_window import make_swatch_icon
        cur = self.window._current_frame_idx
        frame_annos = [a for a in self.annotations if a.frame_idx == cur]
        # Per-class tint for the Class column (matches the spawn buttons).
        class_color = {
            'cell':      QColor('#cccccc'),
            'vessel':    QColor('#9370db'),
            'capillary': QColor('#eb82c8'),
        }
        self.window.list_annotations.blockSignals(True)
        for i in range(self.window.list_annotations.topLevelItemCount()):
            item = self.window.list_annotations.topLevelItem(i)
            if item is None:
                continue
            name = item.text(1)
            anno = next((a for a in frame_annos if a.name == name), None)
            if anno is None:
                continue
            font = item.font(1)
            is_active = (anno == self.active_annotation)
            if is_active and anno.is_locked:
                fg = QColor('#2ecc71'); bold = True
            elif is_active:
                fg = QColor('#ffd700'); bold = True
            elif anno.is_locked:
                fg = QColor('#2ecc71'); bold = False
            elif anno.is_paint_only:
                fg = QColor('#9370db'); bold = False
            else:
                fg = QColor('#cccccc'); bold = False
            font.setBold(bold)
            item.setForeground(1, fg)
            item.setFont(1, font)
            # Class column gets its own subtle tint regardless of state.
            item.setForeground(2, class_color.get(anno.class_type, fg))
            # Keep the swatch in sync — tracking can recolor annotations.
            item.setIcon(0, make_swatch_icon(anno.color))
            # Keep eye/lock glyph columns in sync after Lock All / per-row
            # toggles. _show_frame_annotations sets these on rebuild, but
            # path-aware refreshes go through here.
            item.setText(3, " " if getattr(anno, 'is_hidden', False) else "●")
            item.setText(4, "🔒" if anno.is_locked else "")
        self.window.list_annotations.blockSignals(False)

    # ------------------------------------------------------------------
    # DISPLAY CONTROLS
    # ------------------------------------------------------------------
    def _on_level_slider(self, _value=None):
        lo = self.window.slider_min.value()
        hi = self.window.slider_max.value()
        if lo >= hi:
            hi = lo + 1
            self.window.slider_max.blockSignals(True)
            self.window.slider_max.setValue(hi)
            self.window.slider_max.blockSignals(False)
        self.window._apply_levels(lo, hi)

    def _auto_levels(self):
        self.window._init_level_sliders()

    def _on_colormap_changed(self, _name=None):
        if not self.window.video_data:
            return
        self.window.display_frame(self.window._current_frame_idx)

    # ------------------------------------------------------------------
    # SEG EDITING — mode, brush, fill, save
    # ------------------------------------------------------------------
    def _on_seg_mode_changed(self, mode_id):
        modes = {0: 'select', 1: 'paint', 2: 'erase'}
        new_mode = modes.get(mode_id, 'select')
        if new_mode == 'select' and self._seg_edit_mode != 'select':
            # Leaving brush mode → auto-fit bbox
            self._maybe_fit_bbox(self.active_annotation)
        self._seg_edit_mode = new_mode
        self._set_roi_interactivity(new_mode == 'select')
        self._update_brush_cursor_visibility()

    def _set_seg_mode(self, mode):
        if mode == 'select' and self._seg_edit_mode != 'select':
            # Leaving brush mode → auto-fit bbox
            self._maybe_fit_bbox(self.active_annotation)
        self._seg_edit_mode = mode
        btns = {'select': self.window.btn_mode_select,
                'paint': self.window.btn_mode_paint,
                'erase': self.window.btn_mode_erase}
        if mode in btns:
            btns[mode].setChecked(True)
        self._set_roi_interactivity(mode == 'select')
        self._update_brush_cursor_visibility()

    def _set_roi_interactivity(self, enabled):
        """Enable/disable ROI dragging for current frame's annotations."""
        for anno in self._get_frame_annotations():
            if not anno.is_locked:
                anno.roi.translatable = enabled
                if enabled:
                    anno.roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
                else:
                    anno.roi.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def _on_brush_size_changed(self, val):
        self.window.lbl_brush_size.setText(str(val))
        self._update_brush_cursor_size()

    def _connect_brush_events(self):
        from PyQt6.QtCore import QObject as _QObj
        from PyQt6.QtCore import QEvent

        vb = self.window.view_frame.getView()
        scene = vb.scene()

        # Use SignalProxy for mouse-move (brush cursor + drag painting)
        self._brush_proxy = pg.SignalProxy(
            scene.sigMouseMoved, rateLimit=60, slot=self._on_brush_move)

        # Event filter for proper press/release detection
        class _BrushFilter(_QObj):
            def __init__(self, ctrl, parent=None):
                super().__init__(parent)
                self.ctrl = ctrl

            def eventFilter(self, obj, event):
                if self.ctrl._seg_edit_mode == 'select':
                    return False
                etype = event.type()
                if etype == QEvent.Type.GraphicsSceneMousePress:
                    if event.button() == Qt.MouseButton.LeftButton:
                        # Snapshot the active anno's class layer before the
                        # stroke begins so undo/redo restores the right layer.
                        seg = self.ctrl.window.seg_data
                        anno = self.ctrl.active_annotation
                        if seg is not None and anno is not None:
                            fi = self.ctrl.window._current_frame_idx
                            ct = anno.class_type
                            self.ctrl._brush_mask_snapshot = \
                                seg.get_layer(ct)[fi].copy()
                            self.ctrl._brush_snapshot_class = ct
                        self.ctrl._is_painting = True
                        vb = self.ctrl.window.view_frame.getView()
                        pos = vb.mapSceneToView(event.scenePos())
                        self.ctrl._apply_brush(pos.x(), pos.y())
                        event.accept()
                        return True
                elif etype == QEvent.Type.GraphicsSceneMouseRelease:
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.ctrl._is_painting = False
                        seg = self.ctrl.window.seg_data
                        snap = self.ctrl._brush_mask_snapshot
                        ct = getattr(self.ctrl, '_brush_snapshot_class', 'cell')
                        if seg is not None and snap is not None:
                            fi = self.ctrl.window._current_frame_idx
                            new_mask = seg.get_layer(ct)[fi].copy()
                            if not np.array_equal(snap, new_mask):
                                self.ctrl._undo_stack.push(
                                    BrushStrokeCmd(self.ctrl, fi, snap, new_mask,
                                                   class_type=ct))
                                self.ctrl._mark_seg_dirty()
                                # A paint-only entry that was a phantom
                                # (hidden because no pixels here) becomes
                                # real the moment we paint into it — and
                                # vice-versa for an erase that wipes the
                                # last pixel. Refresh the list so it
                                # reflects the new state immediately.
                                if ct in ('vessel', 'capillary'):
                                    self.ctrl._show_frame_annotations(fi)
                                    self.ctrl._update_stats()
                            self.ctrl._brush_mask_snapshot = None
                            self.ctrl._brush_snapshot_class = None
                        return True
                return False

        self._brush_filter = _BrushFilter(self)
        scene.installEventFilter(self._brush_filter)

    def _on_brush_move(self, args):
        pos_scene = args[0]
        vb = self.window.view_frame.getView()
        pos = vb.mapSceneToView(pos_scene)

        # Update brush cursor position
        if self._brush_cursor is not None and self._seg_edit_mode != 'select':
            r = self._get_brush_radius_in_seg()
            # Brush cursor is in video coords
            scale = self._seg_scale()
            if scale:
                r_vid = r / scale[0]  # approx
            else:
                r_vid = r
            self._brush_cursor.setData(
                [pos.x()], [pos.y()],
                symbolSize=r_vid * 2)

        if self._is_painting:
            self._apply_brush(pos.x(), pos.y())

    def _apply_brush(self, vx, vy):
        """Paint or erase at video coordinates (vx, vy)."""
        seg = self.window.seg_data
        if seg is None:
            return
        anno = self.active_annotation
        if anno is None or anno.instance_id is None:
            return

        scale = self._seg_scale()
        if scale is None:
            return
        sx, sy = scale
        # Convert video coords to seg coords
        cx = vx * sx
        cy = vy * sy
        r = self._get_brush_radius_in_seg()
        frame = self.window._current_frame_idx

        # Each class has its own (T, H, W) layer — cells, vessels, and
        # capillaries are stored independently and can occupy the same
        # pixel. Force only affects within-class overwrite.
        force = self.window.btn_force_paint.isChecked()
        if self._seg_edit_mode == 'paint':
            seg.paint_circle(frame, anno.instance_id, cx, cy, r,
                              force=force, class_type=anno.class_type)
            anno._seg_dirty = True
        elif self._seg_edit_mode == 'erase':
            seg.erase_circle(frame, anno.instance_id, cx, cy, r,
                              class_type=anno.class_type)
            anno._seg_dirty = True

        self.window._update_seg_overlay()

    def _get_brush_radius_in_seg(self):
        """Return brush radius in seg-map pixel coordinates."""
        return self.window.slider_brush_size.value()

    def _update_brush_cursor_visibility(self):
        vb = self.window.view_frame.getView()
        if self._seg_edit_mode != 'select':
            if self._brush_cursor is None:
                self._brush_cursor = pg.ScatterPlotItem(
                    [], [], symbol='o',
                    pen=pg.mkPen('c', width=1),
                    brush=pg.mkBrush(None),
                    pxMode=False)
                self._brush_cursor.setZValue(100)
                vb.addItem(self._brush_cursor)
            self._brush_cursor.setVisible(True)
            self._update_brush_cursor_size()
        else:
            if self._brush_cursor is not None:
                self._brush_cursor.setVisible(False)

    def _update_brush_cursor_size(self):
        if self._brush_cursor is None:
            return
        r = self._get_brush_radius_in_seg()
        scale = self._seg_scale()
        if scale:
            r_vid = r / scale[0]
        else:
            r_vid = r
        # Update size of existing data point
        data = self._brush_cursor.data
        if data is not None and len(data) > 0:
            self._brush_cursor.setSize(r_vid * 2)

    def fill_bbox_cmd(self):
        """Fill the selected annotation's bbox region as a seg instance."""
        seg = self.window.seg_data
        anno = self.active_annotation
        if seg is None or anno is None:
            return

        scale = self._seg_scale()
        if scale is None:
            return
        sx, sy = scale

        # Assign an instance ID if the annotation doesn't have one — pulled
        # from the annotation's own class layer namespace.
        if anno.instance_id is None:
            new_id = seg.next_instance_id(class_type=anno.class_type)
            anno.instance_id = new_id
            color = seg.register_instance_color(new_id, class_type=anno.class_type)
            anno.color = color
            anno.update_visuals()

        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        force = self.window.btn_force_paint.isChecked()
        seg.fill_bbox(anno.frame_idx, anno.instance_id,
                      x * sx, y * sy, w * sx, h * sy,
                      force=force, class_type=anno.class_type)
        anno._seg_dirty = True
        self.window._update_seg_overlay()

    def save_seg_map(self):
        """Save the project — 3 per-class TIFs + Meta.json + project.json.

        All artifacts land in the per-video output folder resolved from
        QSettings (default: ``<video_dir>/out/<stem>/``). Writes are
        atomic; existing files are backed up to ``<file>.bak`` before
        overwrite. Empty class layers are skipped.
        """
        from core import sidecar, project_io

        seg = self.window.seg_data
        if seg is None:
            QMessageBox.warning(self.window, "Save Seg",
                                "No segmentation data loaded.")
            return

        source = seg.filepath or self.window._current_file
        if not source:
            QMessageBox.warning(self.window, "Save Seg",
                                "Open an image/video first so we know where "
                                "to save the masks.")
            return

        out_folder = self._resolve_out_folder(source)
        project_io.ensure_dir(out_folder)

        written = []
        try:
            for ct, fname in project_io.CLASS_MASK_FILES.items():
                layer = seg.get_layer(ct)
                if layer is None or not layer.any():
                    continue
                target = os.path.join(out_folder, fname)
                project_io.atomic_write_tif(target, layer)
                written.append(target)

            meta_path = os.path.join(out_folder, project_io.FILE_META)
            meta = sidecar.collect_meta_from_annotations(self.annotations)
            sidecar.save_meta(meta, meta_path)

            project_io.write_project_manifest(
                out_folder,
                source_video_path=source,
                frame_count=int(seg.num_frames),
                frame_size=(int(seg.height), int(seg.width)),
                class_counts=self._class_counts_for_manifest(),
            )
        except Exception as e:
            QMessageBox.critical(self.window, "Error", str(e))
            return

        # Mark the seg layers as clean so the smart autosave knows it
        # doesn't need to re-flush masks until something changes again.
        self._mark_seg_clean()

        if not written:
            QMessageBox.information(
                self.window, "Save Seg",
                f"No mask layers had pixels — wrote only Meta.json + "
                f"project.json:\n{out_folder}")
            return

        lines = "\n".join(f"  {os.path.basename(p)}" for p in written)
        QMessageBox.information(
            self.window, "Save Seg",
            f"Saved into {out_folder}:\n{lines}\n"
            f"  {project_io.FILE_META}\n"
            f"  {project_io.FILE_PROJECT}")

    def propagate_vein_mask(self):
        """Copy the current frame's painted mask pixels for the active annotation
        to all other frames that share the same annotation (same instance_id).

        Frames that already have pixels for this instance are skipped unless the
        user explicitly confirms overwriting — so per-frame manual edits are
        always preserved by default.

        Only paint-only classes (vessel, capillary) are propagatable. Cells
        are per-frame identities — a Cell_5 on frame 0 is a different cell
        than a Cell_5 on frame 3 (tracking happens later, not here).
        """
        anno = self.active_annotation
        seg = self.window.seg_data
        if anno is None or anno.instance_id is None:
            QMessageBox.information(
                self.window, "Propagate Mask",
                "Select an annotation with a segmentation instance first.")
            return
        if not anno.is_paint_only:
            QMessageBox.information(
                self.window, "Propagate Mask",
                f"Propagate Mask only applies to vessels and capillaries — "
                f"not cells.\n\n"
                f"'{anno.name}' is a {anno.class_type}, which is a per-frame "
                f"identity. Tracking cells across frames is a Phase 5 feature.")
            return
        if seg is None:
            QMessageBox.information(
                self.window, "Propagate Mask",
                "No segmentation data loaded.")
            return

        source_frame = self.window._current_frame_idx
        ct = anno.class_type
        layer = seg.get_layer(ct)
        if not np.any(layer[source_frame] == anno.instance_id):
            QMessageBox.information(
                self.window, "Propagate Mask",
                "No painted pixels for this annotation on the current frame.\n\n"
                "Paint the mask here first, then use Propagate Mask to copy it "
                "to all other frames.")
            return

        # Collect all frames that have this annotation. Identity is
        # (class_type, instance_id) — without the class filter a
        # vessel and a capillary that share an iid get merged here.
        anno_frames = sorted({a.frame_idx for a in self.annotations
                              if a.instance_id == anno.instance_id
                              and a.class_type == ct})
        total_frames = self.window.video_data.num_frames

        # If the annotation only exists on the current frame, auto-expand
        # it to every other frame before propagating. This matches the
        # user expectation that "Propagate Mask" really does the whole
        # job from one click. The newly spawned annotations and the
        # propagated pixels go into one undo step.
        spawned = []
        if anno_frames == [source_frame]:
            for fi in range(total_frames):
                if fi == source_frame:
                    continue
                new_anno = Annotation2D(
                    anno.name, self.window.view_frame, self,
                    start_pos=(0, 0),
                    start_size=(1, 1),
                    shape_mode='rect',
                    frame_idx=fi,
                    instance_id=anno.instance_id,
                    color=anno.color,
                    class_type=anno.class_type,
                )
                new_anno.sig_clicked.connect(self.select_annotation)
                new_anno.sig_updated.connect(self._on_anno_updated)
                self.annotations.append(new_anno)
                new_anno.roi.setVisible(False)
                spawned.append(new_anno)
            anno_frames = sorted(anno_frames + [a.frame_idx for a in spawned])
        target_frames = [f for f in anno_frames if f != source_frame]

        if not target_frames:
            # Nothing to do — single-frame stack.
            return

        # Determine which targets already have pixels
        already_painted = [f for f in target_frames
                           if np.any(layer[f] == anno.instance_id)]
        empty_targets   = [f for f in target_frames if f not in already_painted]

        overwrite = False
        if already_painted:
            reply = QMessageBox.question(
                self.window, "Propagate Mask",
                f"{len(empty_targets)} frame(s) are empty and will be filled.\n"
                f"{len(already_painted)} frame(s) already have pixels.\n\n"
                "Overwrite those frames too?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            overwrite = (reply == QMessageBox.StandardButton.Yes)

        effective_targets = target_frames if overwrite else empty_targets
        if not effective_targets:
            return

        # Snapshot for undo
        old_masks = {fi: layer[fi].copy() for fi in effective_targets}
        n_updated = seg.propagate_instance_mask(
            anno.instance_id, source_frame, effective_targets,
            overwrite=overwrite, class_type=ct)
        new_masks = {fi: layer[fi].copy() for fi in effective_targets}

        if n_updated > 0:
            if spawned:
                # Group spawn + propagate into one undo step.
                self._undo_stack.push(
                    PropagateWithSpawnCmd(
                        self, spawned, old_masks, new_masks, class_type=ct))
            else:
                self._undo_stack.push(
                    PropagateMaskCmd(self, old_masks, new_masks, class_type=ct))
        elif spawned:
            # No pixels propagated but we still spawned annos — keep them
            # undoable as a batch.
            self._undo_stack.push(AddAnnotationBatchCmd(self, spawned))

        # Make sure the new annotations show up in the list immediately.
        if spawned:
            self._show_frame_annotations(self.window._current_frame_idx)

        self.window._update_seg_overlay()
        # Brief status feedback without a blocking dialog
        self.window.lbl_stats.setText(
            f"Propagated to {n_updated} frame(s)  \u2014  Ctrl+Z to undo")
        QTimer.singleShot(4000, self._update_stats)

    # ------------------------------------------------------------------
    # SEGMENTATION LOADING  →  BBOX ANNOTATIONS
    # ------------------------------------------------------------------
    # Distinct colours assigned to instances (not from the seg-map palette).
    _INSTANCE_COLORS = [
        (255, 80, 80),   (80, 255, 80),   (80, 160, 255),  (255, 255, 80),
        (255, 80, 255),  (80, 255, 255),  (255, 160, 80),  (160, 80, 255),
        (80, 255, 160),  (255, 128, 128), (128, 255, 128), (128, 128, 255),
        (220, 220, 80),  (220, 80, 220),  (80, 220, 220),  (255, 200, 120),
        (120, 200, 255), (200, 255, 120), (240, 128, 180), (128, 240, 180),
    ]

    def load_project_folder(self):
        """User picks an output folder; we load every class TIF + meta
        from it as a single project. Replaces the current session.

        Defaults the dialog to the current video's resolved out folder,
        so the user just hits Enter in the common case.
        """
        from core import project_io

        start = (self._resolve_out_folder()
                  or (os.path.dirname(self.window._current_file)
                       if self.window._current_file else os.getcwd()))
        # If start doesn't exist yet, walk up to the nearest real folder.
        while start and not os.path.isdir(start):
            parent = os.path.dirname(start)
            if parent == start:
                start = os.getcwd()
                break
            start = parent

        folder = QFileDialog.getExistingDirectory(
            self.window, "Load project folder", start)
        if not folder:
            return

        summary = project_io.session_summary(folder)
        if summary is None or not summary.get('has_masks'):
            QMessageBox.warning(
                self.window, "Load Project",
                f"No mask TIFs found in:\n{folder}\n\n"
                f"Expected at least one of "
                f"{', '.join(project_io.CLASS_MASK_FILES.values())}.")
            return

        if self.annotations:
            reply = QMessageBox.question(
                self.window, "Replace current session?",
                f"Loading this project will replace the current session "
                f"(annotations, masks, meta).\n\n"
                f"  Folder:  {folder}\n"
                f"  Saved:   {summary.get('updated_at', '?')}\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            self._load_session_from_out_folder(folder)
        except Exception as e:
            log_error('controller.load_project', 'load failed', exc=e)
            QMessageBox.critical(
                self.window, "Load Project",
                f"Could not load project:\n\n{type(e).__name__}: {e}")
            return

        QMessageBox.information(
            self.window, "Load Project",
            f"Loaded:\n{folder}")

    def load_single_class_tif(self):
        """Import a single TIF into one chosen class layer, merging it
        into the current session without touching the other classes.
        """
        from core import mask_io, project_io
        from core.volume_data import SegmentationData

        start_dir = (self._resolve_out_folder()
                      or (os.path.dirname(self.window._current_file)
                           if self.window._current_file else os.getcwd()))
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Load single-class mask TIF", start_dir,
            "Instance mask TIF (*.tif *.tiff);;All Files (*)")
        if not path:
            return

        from PyQt6.QtWidgets import QInputDialog
        choices = ['cell', 'vessel', 'capillary']
        # Pre-select based on filename if possible.
        base = os.path.basename(path).lower()
        default_idx = 0
        if 'vessel' in base:
            default_idx = 1
        elif 'capill' in base:
            default_idx = 2
        ct, ok = QInputDialog.getItem(
            self.window, "Load Class",
            f"Import {os.path.basename(path)} as which class?",
            [c.capitalize() for c in choices], default_idx, False)
        if not ok:
            return
        ct = ct.lower()

        try:
            arr = mask_io.load_mask_tif(path).astype(np.int32)
        except Exception as e:
            QMessageBox.critical(self.window, "Load Class",
                                  f"Failed to read TIF:\n\n{e}")
            return

        T, H, W = arr.shape
        seg = self.window.seg_data
        if seg is None:
            seg = SegmentationData.empty(W, H, T)
            seg.filepath = path
            self.window.seg_data = seg
            self.window._seg_visible = True
        else:
            if (T, H, W) != (seg.num_frames, seg.height, seg.width):
                QMessageBox.critical(
                    self.window, "Load Class",
                    f"Shape mismatch with current session:\n"
                    f"  current: {(seg.num_frames, seg.height, seg.width)}\n"
                    f"  picked:  {(T, H, W)}\n\n"
                    f"Refusing to load a layer that doesn't fit.")
                return

        # Confirm overwrite if the target class already has content.
        existing_layer = seg.get_layer(ct)
        if existing_layer is not None and np.any(existing_layer):
            reply = QMessageBox.question(
                self.window, "Overwrite class?",
                f"The {ct} layer already has painted pixels. Replace it "
                f"with the contents of {os.path.basename(path)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Wipe the target class's annotations + colors + pixels, then
        # rebuild from the freshly-loaded layer.
        to_drop = [a for a in self.annotations if a.class_type == ct]
        for anno in to_drop:
            try:
                anno.delete_ui()
            except Exception:
                pass
            self.annotations.remove(anno)
        if self.active_annotation in to_drop:
            self.active_annotation = None
        seg.get_colors(ct).clear()
        seg.set_layer(ct, arr)
        for iid in np.unique(arr):
            if iid == 0:
                continue
            seg.register_instance_color(int(iid), class_type=ct)

        # Rebuild annotations for the loaded class only.
        if ct == 'cell':
            for frame_idx in range(seg.num_frames):
                bboxes = seg.get_all_bboxes(frame_idx, class_type='cell')
                for (stair_id, blob_idx), (x0, y0, w, h) in bboxes.items():
                    name = (f"Cell_{stair_id}" if blob_idx == 0
                             else f"Cell_{stair_id}_{blob_idx}")
                    anno = Annotation2D(
                        name, self.window.view_frame, self,
                        start_pos=(x0, y0), start_size=(w, h),
                        shape_mode='rect', frame_idx=frame_idx,
                        instance_id=stair_id,
                        color=seg.get_colors('cell').get(
                            int(stair_id), (255, 80, 80)),
                    )
                    anno.sig_clicked.connect(self.select_annotation)
                    anno.sig_updated.connect(self._on_anno_updated)
                    self.annotations.append(anno)
                    anno.roi.setVisible(False)
        else:
            label = 'Vessel' if ct == 'vessel' else 'Capillary'
            colors = seg.get_colors(ct)
            for fi in range(seg.num_frames):
                for iid in np.unique(arr[fi]):
                    if iid == 0:
                        continue
                    iid = int(iid)
                    anno = Annotation2D(
                        f"{label}_{iid}", self.window.view_frame, self,
                        start_pos=(0, 0), start_size=(1, 1),
                        shape_mode='rect', frame_idx=fi,
                        instance_id=iid,
                        color=colors.get(iid, (147, 112, 219)),
                        class_type=ct)
                    anno.sig_clicked.connect(self.select_annotation)
                    anno.sig_updated.connect(self._on_anno_updated)
                    self.annotations.append(anno)

        self._normalize_anno_names_and_colors()
        self._mark_seg_dirty()  # imported pixels need a save
        cur = self.window._current_frame_idx
        self._show_frame_annotations(cur)
        self.window._update_seg_overlay()
        self._update_stats()
        log('controller.load_class', 'merged single class',
            class_type=ct, path=path,
            n_instances=int(len(np.unique(arr)) - 1))
        QMessageBox.information(
            self.window, "Load Class",
            f"Imported {ct} layer from:\n{os.path.basename(path)}")

    def load_segmentation(self):
        from core.volume_data import SegmentationData
        from core import mask_io

        # Default search dir = location of the open image, if any.
        start_dir = (os.path.dirname(self.window._current_file)
                     if self.window._current_file else os.getcwd())
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Load Segmentation Map",
            start_dir,
            "Instance mask TIF (*.tif *.tiff);;Video Files (*.avi *.mp4);;All Files (*)")
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in ('.tif', '.tiff'):
                # Three layouts to detect:
                #   1. Project out folder containing Cells.tif / Vessels.tif / Capillaries.tif
                #   2. Legacy next-to-video {stem}_Cells.tif etc.
                #   3. Standalone TIF — treat as a cell layer.
                from core import project_io
                picked_dir = os.path.dirname(os.path.abspath(path))
                picked_name = os.path.basename(path)
                if picked_name in project_io.CLASS_MASK_FILES.values():
                    # Inside an out folder — sweep all canonical files there.
                    layers = mask_io.load_multiclass_from_folder(picked_dir)
                else:
                    anchor = path
                    for ct, suffix in mask_io.CLASS_FILENAME_SUFFIX.items():
                        if path.endswith(suffix):
                            anchor = path[:-len(suffix)] + '.x'
                            break
                    if path.endswith('_Masks.tif'):
                        anchor = path[:-len('_Masks.tif')] + '.x'
                    layers = mask_io.load_multiclass_masks(anchor)

                if not layers:
                    # Fall back: treat the picked file alone as the cells layer
                    layers = {'cell': mask_io.load_mask_tif(path)}

                # All layers must share shape; cells (or whichever is
                # present) defines the canvas.
                first_layer = next(iter(layers.values()))
                T, H, W = first_layer.shape

                vd = self.window.video_data
                if vd is not None and (H, W) != (vd.height, vd.width):
                    QMessageBox.warning(
                        self.window, "Load Seg",
                        f"Mask size {(H, W)} does not match image "
                        f"{(vd.height, vd.width)}. Loading anyway — overlay "
                        f"will be resized at display time.")

                seg = SegmentationData.empty(W, H, T)
                seg.filepath = path
                for ct, arr in layers.items():
                    if arr.shape != (T, H, W):
                        raise ValueError(
                            f"{ct} layer shape {arr.shape} does not match "
                            f"cell shape {(T, H, W)}")
                    seg.set_layer(ct, arr.astype(np.int32))
                    for iid in np.unique(arr):
                        if iid == 0:
                            continue
                        seg.register_instance_color(int(iid), class_type=ct)
            else:
                seg = SegmentationData(path)
        except Exception as e:
            QMessageBox.critical(self.window, "Error", str(e))
            return

        self.window.seg_data = seg
        self.window._seg_visible = True

        # Clear existing annotations before importing
        self._clear_all_annotations()

        # Use the stair-based colours already computed by SegmentationData
        stair_color_map = dict(seg.instance_colors)

        # Scale factors if seg map size differs from video
        sx, sy = 1.0, 1.0
        if self.window.video_data:
            if seg.width != self.window.video_data.width:
                sx = self.window.video_data.width / seg.width
            if seg.height != self.window.video_data.height:
                sy = self.window.video_data.height / seg.height

        anno_count = 0
        max_stair = 0
        for frame_idx in range(seg.num_frames):
            bboxes = seg.get_all_bboxes(frame_idx)
            for (stair_id, blob_idx), (x0, y0, w, h) in bboxes.items():
                color = stair_color_map.get(stair_id, (255, 80, 80))
                if blob_idx == 0:
                    name = f"Cell_{stair_id}"
                else:
                    name = f"Cell_{stair_id}_{blob_idx}"
                if stair_id > max_stair:
                    max_stair = stair_id

                # Scale bbox to video coordinates
                ax = x0 * sx
                ay = y0 * sy
                aw = w * sx
                ah = h * sy

                anno = Annotation2D(
                    name, self.window.view_frame, self,
                    start_pos=(ax, ay),
                    start_size=(aw, ah),
                    shape_mode='rect',
                    frame_idx=frame_idx,
                    instance_id=stair_id,
                    color=color,
                )
                anno.sig_clicked.connect(self.select_annotation)
                anno.sig_updated.connect(self._on_anno_updated)
                self.annotations.append(anno)
                # Hide ROI; _show_frame_annotations will reveal current frame
                anno.roi.setVisible(False)
                anno_count += 1

        self.anno_counter = max_stair

        # Build paint-only annotations from the vessel and capillary
        # layers — one entry per (instance_id, frame_idx).
        for ct in ('vessel', 'capillary'):
            layer = seg.get_layer(ct)
            colors = seg.get_colors(ct)
            if layer is None:
                continue
            for fi in range(seg.num_frames):
                frame_ids = np.unique(layer[fi])
                for iid in frame_ids:
                    if iid == 0:
                        continue
                    iid = int(iid)
                    color = colors.get(iid, (147, 112, 219))
                    label = 'Vessel' if ct == 'vessel' else 'Capillary'
                    name = f"{label}_{iid}"
                    anno = Annotation2D(
                        name, self.window.view_frame, self,
                        start_pos=(0, 0),
                        start_size=(1, 1),
                        shape_mode='rect',
                        frame_idx=fi,
                        instance_id=iid,
                        color=color,
                        class_type=ct,
                    )
                    anno.sig_clicked.connect(self.select_annotation)
                    anno.sig_updated.connect(self._on_anno_updated)
                    self.annotations.append(anno)
                    anno_count += 1

        # Apply per-instance metadata from the meta sidecar (if present)
        # so names, class_type, and locked state survive a save/load cycle.
        # The lookup is class-aware so vessels and capillaries with the
        # same iid never adopt each other's records. Look in the new
        # out-folder layout first, then fall back to the legacy sidecar
        # next to the picked TIF.
        from core import sidecar, project_io
        meta = None
        picked_dir = os.path.dirname(os.path.abspath(path))
        picked_meta = os.path.join(picked_dir, project_io.FILE_META)
        if os.path.exists(picked_meta):
            meta = sidecar.load_meta(picked_meta)
        if meta is None:
            meta = sidecar.load_meta(sidecar.meta_path_for(path))
        if meta is not None:
            for anno in self.annotations:
                if anno.instance_id is None:
                    continue
                rec = sidecar.meta_lookup(
                    meta, anno.class_type, int(anno.instance_id))
                if not rec:
                    continue
                # Annotation2D normalizes 'vein' -> 'vessel' in __init__,
                # but here we have an already-constructed object.
                ct = rec.get('class_type', anno.class_type)
                if ct == 'vein':
                    ct = 'vessel'
                if ct in ('cell', 'vessel', 'capillary'):
                    anno.class_type = ct
                if 'name' in rec and rec['name']:
                    anno.name = rec['name']
                if rec.get('locked'):
                    anno.set_locked(True)
                if 'notes' in rec:
                    anno.notes = rec['notes']

        # Repair any stale name/color mismatches against class_type. Old
        # save files predate the per-class instance-ID namespace, so an
        # iid=4 capillary may currently carry a "Cell_4" name and a
        # cell-palette color — fix in place.
        self._normalize_anno_names_and_colors()

        # Show annotations for the current frame
        cur = self.window._current_frame_idx
        self._show_frame_annotations(cur)
        self.window._update_seg_overlay()

        all_stairs = sorted(stair_color_map.keys())
        QMessageBox.information(
            self.window, "Segmentation Loaded",
            f"Created {anno_count} bbox annotations from "
            f"{len(all_stairs)} stairs across {seg.num_frames} frames.")

    # ------------------------------------------------------------------
    # SAM panel helpers (Phase 4.1)
    # ------------------------------------------------------------------
    # Map combo-box index -> (model_type, checkpoint_path | None).
    _SAM_MODEL_CHOICES = [
        ('vit_b', 'sam_hela'),   # fine-tuned default; checkpoint resolved at runtime
        ('vit_b_lm', None),
        ('vit_t', None),
        ('vit_b', None),
        ('vit_l', None),
    ]

    def _on_sam_model_changed(self, idx):
        # Stop any in-flight embed worker — the new service has its own cache.
        self._stop_embed_worker(timeout_ms=5000)
        model_type, hint = self._SAM_MODEL_CHOICES[
            idx if 0 <= idx < len(self._SAM_MODEL_CHOICES) else 0]
        if hint == 'sam_hela':
            ckpt = default_sam_hela_path()
        else:
            ckpt = None
        self.sam_service = SamService(model_type=model_type, checkpoint_path=ckpt)
        self._refresh_sam_status()
        # If we have an image loaded, kick off precompute for the current
        # frame with the new model (the previous model's embeddings are
        # in a different cache dir so this is fresh work).
        self.on_image_loaded()

    # ------------------------------------------------------------------
    # Embedding precompute (Phase 1a) — async background worker
    # ------------------------------------------------------------------
    def _stop_embed_worker(self, timeout_ms=3000):
        """Politely stop any running embed worker and wait for it.

        Used before any main-thread SAM call (predictor isn't thread-safe)
        and before starting a new worker (only one alive at a time).
        """
        w = self._embed_worker
        if w is None:
            return
        if not w.isRunning():
            self._embed_worker = None
            return
        w.request_stop()
        if not w.wait(timeout_ms):
            log_error('controller.sam',
                      'embed worker did not stop — forcing terminate')
            w.terminate()
            w.wait(500)
        self._embed_worker = None

    def _frames_to_compute(self, frame_indices):
        """Skip frames whose embedding is already cached."""
        img = self.window._current_file
        if img is None:
            return []
        out = []
        for fi in frame_indices:
            if not self.sam_service.has_cached_embedding(img, fi):
                out.append((int(fi), self.window.video_data.get_frame(int(fi))))
        return out

    def _start_embed_worker(self, frame_indices, *, label="", interactive=True):
        """Kick off embedding precompute in the background.

        interactive=True (default) → show a modal progress dialog that
        blocks the rest of the UI while the model runs. Use this for
        explicit user triggers (Precompute All, model swap, the first
        embedding after open).

        interactive=False → no dialog, just the status-bar text. Used by
        the silent auto-precompute on frame change so casual scrubbing
        doesn't get interrupted by modal pop-ups.
        """
        if not SamService.available():
            return
        if self.window.video_data is None or self.window._current_file is None:
            return
        frames = self._frames_to_compute(frame_indices)
        if not frames:
            return  # everything already cached
        self._stop_embed_worker(timeout_ms=5000)
        self._embed_worker = EmbeddingPrecomputeWorker(
            self.sam_service, self.window._current_file, frames,
            parent=self.window)
        self._embed_worker.progress.connect(self._on_embed_progress)
        self._embed_worker.frame_done.connect(self._on_embed_frame_done)
        self._embed_worker.error.connect(self._on_embed_error)
        self._embed_worker.finished_ok.connect(self._on_embed_finished_ok)
        self._embed_label = label or f"frame {frames[0][0]}"
        self._embed_total = len(frames)
        self.window.lbl_sam_status.setText(
            f"Precomputing {self._embed_label} (0/{self._embed_total})…")
        self.window.btn_sam_precompute.setEnabled(False)
        if interactive:
            self._show_embed_dialog(self._embed_total, self._embed_label)
        self._embed_worker.start()
        log('controller.sam', 'embed worker started',
            n_frames=len(frames), label=self._embed_label,
            interactive=interactive)

    # --- Modal progress dialog --------------------------------------------
    def _show_embed_dialog(self, total, label):
        """Open an application-modal progress dialog for the active embed
        run. Closes itself in _on_embed_finished_ok / _on_embed_error."""
        from PyQt6.QtWidgets import QProgressDialog
        if self._embed_dialog is not None:
            try:
                self._embed_dialog.close()
            except RuntimeError:
                pass
            self._embed_dialog = None
        dlg = QProgressDialog(
            f"Computing SAM embeddings for {label}…\n"
            "The rest of the UI is paused until this finishes.",
            "Cancel", 0, max(1, total), self.window)
        dlg.setWindowTitle("SAM model running")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)   # show immediately, no auto-hide grace
        dlg.setValue(0)
        dlg.setAutoReset(False)
        dlg.setAutoClose(False)
        dlg.canceled.connect(self._on_embed_dialog_cancel)
        self._embed_dialog = dlg
        dlg.show()

    def _close_embed_dialog(self):
        dlg = self._embed_dialog
        self._embed_dialog = None
        if dlg is None:
            return
        try:
            dlg.close()
        except RuntimeError:
            pass

    def _on_embed_dialog_cancel(self):
        """Cancel button → ask the worker to stop between frames."""
        w = self._embed_worker
        if w is not None and w.isRunning():
            w.request_stop()
        self._close_embed_dialog()

    def _on_embed_progress(self, done, total):
        self.window.lbl_sam_status.setText(
            f"Precomputing {self._embed_label} ({done}/{total})…")
        dlg = self._embed_dialog
        if dlg is not None:
            dlg.setMaximum(max(1, total))
            dlg.setValue(done)
            dlg.setLabelText(
                f"Computing SAM embedding {done}/{total}\n"
                f"({self._embed_label})\n"
                "The rest of the UI is paused until this finishes.")

    def _on_embed_frame_done(self, frame_idx):
        log('controller.sam', 'frame embedding cached', frame_idx=frame_idx)

    def _on_embed_error(self, msg):
        log_error('controller.sam', f'embed worker error: {msg}')
        self.window.lbl_sam_status.setText(f"Embedding error: {msg}")
        self.window.btn_sam_precompute.setEnabled(True)
        self._close_embed_dialog()
        self._drain_pending_embed()

    def _on_embed_finished_ok(self):
        self.window.lbl_sam_status.setText(
            f"Embeddings ready · {self._embed_total} frame(s) cached.")
        self.window.btn_sam_precompute.setEnabled(True)
        self._close_embed_dialog()
        self._refresh_sam_status_brief()
        self._drain_pending_embed()

    def _drain_pending_embed(self):
        """If a frame was requested while the worker was busy, start it now."""
        fi = self._embed_pending_frame
        self._embed_pending_frame = None
        if fi is None:
            return
        # Skip if the user has navigated past it and it's already cached
        # (typical for stair-step scrubbing).
        if self.window._current_file is None:
            return
        if self.sam_service.has_cached_embedding(
                self.window._current_file, fi):
            return
        # Frame-change drain remains silent — same UX as the original request.
        self._start_embed_worker(
            [fi], label=f"frame {fi}", interactive=False)

    def _refresh_sam_status_brief(self):
        """Restore the standard 'model: …' status after a short delay so
        the success message stays visible briefly."""
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2500, self._refresh_sam_status)

    def on_image_loaded(self):
        """Called by main.py and the model selector when a fresh image
        is in play.

        Two responsibilities:
          1. Offer to resume the previous session (if one exists in the
             out folder for this video).
          2. Kick off the auto-precompute of the current frame's SAM
             embedding so the first SAM Box click is instant.
        """
        # Resume prompt comes first so the embed worker (if started)
        # only runs against the loaded session.
        self._maybe_prompt_resume()
        if not SamService.available() or self.window.video_data is None:
            return
        if self.window._current_file is None:
            return
        fi = self.window._current_frame_idx
        self._start_embed_worker([fi], label=f"frame {fi}")

    def _maybe_prompt_resume(self):
        """If the current video has a saved session in its out folder,
        ask the user whether to resume it.

        Does nothing when no out folder / no saved files exist, when no
        video is loaded, or when annotations are already in memory (we
        don't want to clobber unsaved work)."""
        from core import project_io
        if self.window._current_file is None:
            return
        if self.annotations:
            return  # work already in progress — don't ask
        out_folder = self._resolve_out_folder()
        summary = project_io.session_summary(out_folder)
        if summary is None or not summary.get('has_masks'):
            # Also consider the legacy next-to-video sidecars: load_segmentation
            # already handles them via the file picker, but for resume we keep
            # it simple — only prompt when the new layout has artifacts.
            return

        manifest = summary.get('manifest') or {}
        counts = manifest.get('class_counts', {})
        when = summary.get('updated_at', '?')
        n_cell = counts.get('cell', '?')
        n_vess = counts.get('vessel', '?')
        n_cap  = counts.get('capillary', '?')

        msg = QMessageBox(self.window)
        msg.setWindowTitle("Resume session?")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(
            f"Found a saved session for this video.\n\n"
            f"  Folder:  {out_folder}\n"
            f"  Saved:   {when}\n"
            f"  Counts:  {n_cell} cells · {n_vess} vessels · {n_cap} capillaries\n")
        msg.setInformativeText(
            "Resume picks up where you left off (masks + names + locks).\n"
            "Start fresh leaves the saved files alone — you can still "
            "save over them later.")
        btn_resume = msg.addButton("Resume", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Start fresh", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_resume)
        msg.exec()
        if msg.clickedButton() is not btn_resume:
            return

        try:
            self._load_session_from_out_folder(out_folder)
        except Exception as e:
            log_error('controller.resume', 'resume load failed', exc=e)
            QMessageBox.critical(
                self.window, "Resume failed",
                f"Could not load saved session:\n\n{type(e).__name__}: {e}")

    def _load_session_from_out_folder(self, out_folder):
        """Programmatic load equivalent to load_segmentation, but
        sourced from an output folder rather than a file picker."""
        from core import mask_io, sidecar, project_io
        from core.volume_data import SegmentationData

        layers = mask_io.load_multiclass_from_folder(out_folder)
        if not layers:
            return  # nothing on disk after all

        first_layer = next(iter(layers.values()))
        T, H, W = first_layer.shape
        seg = SegmentationData.empty(W, H, T)
        seg.filepath = os.path.join(
            out_folder, project_io.CLASS_MASK_FILES.get('cell', 'Cells.tif'))
        for ct, arr in layers.items():
            if arr.shape != (T, H, W):
                raise ValueError(
                    f"{ct} layer shape {arr.shape} does not match "
                    f"cell shape {(T, H, W)}")
            seg.set_layer(ct, arr.astype(np.int32))
            for iid in np.unique(arr):
                if iid == 0:
                    continue
                seg.register_instance_color(int(iid), class_type=ct)

        self.window.seg_data = seg
        self.window._seg_visible = True
        self._clear_all_annotations()

        # Rebuild annotations from each layer the same way load_segmentation
        # does — cells get bboxes (one per blob); vessels & capillaries get
        # paint-only entries per (frame, iid).
        stair_color_map = dict(seg.instance_colors)
        sx = sy = 1.0
        if self.window.video_data:
            if seg.width != self.window.video_data.width:
                sx = self.window.video_data.width / seg.width
            if seg.height != self.window.video_data.height:
                sy = self.window.video_data.height / seg.height

        max_stair = 0
        for frame_idx in range(seg.num_frames):
            bboxes = seg.get_all_bboxes(frame_idx)
            for (stair_id, blob_idx), (x0, y0, w, h) in bboxes.items():
                color = stair_color_map.get(stair_id, (255, 80, 80))
                name = f"Cell_{stair_id}" if blob_idx == 0 \
                       else f"Cell_{stair_id}_{blob_idx}"
                if stair_id > max_stair:
                    max_stair = stair_id
                anno = Annotation2D(
                    name, self.window.view_frame, self,
                    start_pos=(x0 * sx, y0 * sy),
                    start_size=(w * sx, h * sy),
                    shape_mode='rect',
                    frame_idx=frame_idx,
                    instance_id=stair_id,
                    color=color,
                )
                anno.sig_clicked.connect(self.select_annotation)
                anno.sig_updated.connect(self._on_anno_updated)
                self.annotations.append(anno)
                anno.roi.setVisible(False)
        self.anno_counter = max_stair

        for ct in ('vessel', 'capillary'):
            layer = seg.get_layer(ct)
            if layer is None:
                continue
            colors = seg.get_colors(ct)
            for fi in range(seg.num_frames):
                for iid in np.unique(layer[fi]):
                    if iid == 0:
                        continue
                    iid = int(iid)
                    color = colors.get(iid, (147, 112, 219))
                    label = 'Vessel' if ct == 'vessel' else 'Capillary'
                    anno = Annotation2D(
                        f"{label}_{iid}", self.window.view_frame, self,
                        start_pos=(0, 0), start_size=(1, 1),
                        shape_mode='rect', frame_idx=fi,
                        instance_id=iid, color=color, class_type=ct)
                    anno.sig_clicked.connect(self.select_annotation)
                    anno.sig_updated.connect(self._on_anno_updated)
                    self.annotations.append(anno)

        # Meta sidecar in the out folder (class-aware lookups).
        meta_path = os.path.join(out_folder, project_io.FILE_META)
        meta = sidecar.load_meta(meta_path)
        if meta is not None:
            for anno in self.annotations:
                if anno.instance_id is None:
                    continue
                rec = sidecar.meta_lookup(
                    meta, anno.class_type, int(anno.instance_id))
                if not rec:
                    continue
                ct = rec.get('class_type', anno.class_type)
                if ct == 'vein':
                    ct = 'vessel'
                if ct in ('cell', 'vessel', 'capillary'):
                    anno.class_type = ct
                if rec.get('name'):
                    anno.name = rec['name']
                if rec.get('locked'):
                    anno.set_locked(True)
                if 'notes' in rec:
                    anno.notes = rec['notes']

        self._normalize_anno_names_and_colors()
        self._mark_seg_clean()
        # Preserve the manifest timestamp on the status label so the
        # user sees "Saved 2 days ago" rather than "Saved 0s ago" right
        # after a resume.
        manifest = project_io.read_project_manifest(out_folder) or {}
        when = manifest.get('updated_at')
        if when:
            try:
                import datetime as _dt
                self._last_save_at = _dt.datetime.fromisoformat(when).timestamp()
                self._refresh_save_status()
            except (TypeError, ValueError):
                pass
        cur = self.window._current_frame_idx
        self._show_frame_annotations(cur)
        self.window._update_seg_overlay()
        self._update_stats()
        log('controller.resume', 'session loaded',
            out_folder=out_folder, n_annos=len(self.annotations))

    def _on_frame_changed_embed(self, frame_idx):
        """Hook on state.frame_changed — precompute the new frame in the
        background (skipped if already cached).

        Never blocks: if a worker is already busy, the frame is queued and
        picked up the moment that worker finishes. This avoids freezing the
        UI when the user scrubs through frames faster than embeddings
        compute (~3s on cold MPS)."""
        if not SamService.available() or self.window.video_data is None:
            return
        if self.window._current_file is None:
            return
        if self.sam_service.has_cached_embedding(
                self.window._current_file, frame_idx):
            self._embed_pending_frame = None
            return
        # If a worker is already running, just remember the most recent
        # frame request and return immediately.
        w = self._embed_worker
        if w is not None and w.isRunning():
            self._embed_pending_frame = int(frame_idx)
            return
        self._embed_pending_frame = None
        # Auto-precompute on frame change runs silently — scrubbing must
        # not pop modals on every step.
        self._start_embed_worker(
            [frame_idx], label=f"frame {frame_idx}", interactive=False)

    def precompute_all_frames(self):
        """Triggered by the 'Precompute embeddings' button.

        Walks every frame in the stack (skipping cached ones), encoding
        each in the background. Status line shows progress; the button
        re-enables when done.
        """
        if not SamService.available():
            QMessageBox.critical(self.window, "Error", "MicroSAM is not installed.")
            return
        if self.window.video_data is None:
            return
        n = self.window.video_data.num_frames
        # Warn on very large stacks since each is ~4 MB on disk.
        if n > 200:
            reply = QMessageBox.question(
                self.window, "Precompute all frames",
                f"This will encode {n} frames (~{n * 4} MB on disk and "
                f"roughly {n * 2}s of compute on MPS). Continue?")
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._start_embed_worker(
            list(range(n)), label=f"all {n} frames")

    # ------------------------------------------------------------------
    # Tracking panel helpers
    # ------------------------------------------------------------------
    def _on_tracker_changed(self, name):
        """Combobox selection -> rebuild service and settings widgets."""
        try:
            self.tracker_service = make_tracker(name)
        except Exception as e:
            log_error('controller.tracking', 'tracker swap failed', exc=e)
            return
        self._rebuild_tracker_settings()
        log('controller.tracking', 'tracker selected', name=name,
            settings=dict(self.tracker_service.settings))

    def _rebuild_tracker_settings(self):
        """Repopulate the dynamic settings panel for the active tracker."""
        from PyQt6.QtWidgets import (QSpinBox, QDoubleSpinBox, QCheckBox,
                                     QComboBox, QLabel)
        layout = self.window.tracker_settings_layout
        # Clear existing widgets.
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # Build new ones.
        for spec in self.tracker_service.setting_specs():
            widget = self._make_setting_widget(spec)
            label = QLabel(spec.label)
            label.setToolTip(spec.tooltip)
            layout.addRow(label, widget)

    def _make_setting_widget(self, spec):
        """Map a SettingSpec to a Qt widget bound to tracker_service.settings."""
        from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLabel
        cur = self.tracker_service.settings.get(spec.key, spec.default)
        if spec.kind == 'int':
            w = QSpinBox()
            if spec.min is not None:
                w.setMinimum(int(spec.min))
            if spec.max is not None:
                w.setMaximum(int(spec.max))
            w.setValue(int(cur))
            w.setToolTip(spec.tooltip)
            w.valueChanged.connect(
                lambda v, k=spec.key: self.tracker_service.update_setting(k, int(v)))
            return w
        if spec.kind == 'float':
            w = QDoubleSpinBox()
            if spec.min is not None:
                w.setMinimum(float(spec.min))
            if spec.max is not None:
                w.setMaximum(float(spec.max))
            if spec.step:
                w.setSingleStep(float(spec.step))
            w.setValue(float(cur))
            w.setToolTip(spec.tooltip)
            w.valueChanged.connect(
                lambda v, k=spec.key: self.tracker_service.update_setting(k, float(v)))
            return w
        if spec.kind == 'bool':
            w = QCheckBox()
            w.setChecked(bool(cur))
            w.setToolTip(spec.tooltip)
            w.toggled.connect(
                lambda v, k=spec.key: self.tracker_service.update_setting(k, bool(v)))
            return w
        if spec.kind == 'choice':
            w = QComboBox()
            for c in (spec.choices or []):
                w.addItem(c)
            w.setCurrentText(str(cur))
            w.setToolTip(spec.tooltip)
            w.currentTextChanged.connect(
                lambda v, k=spec.key: self.tracker_service.update_setting(k, v))
            return w
        return QLabel(f"(unsupported: {spec.kind})")

    def run_tracking_now(self):
        """Invoke the active tracker on the current segmentation.

        Tracking is cells-only by design — vessels and capillaries don't
        move across frames and aren't a meaningful track target — so the
        gate here intentionally only checks the cell layer."""
        seg = self.window.seg_data
        if seg is None or not np.any(seg.get_layer('cell')):
            QMessageBox.information(
                self.window, "Run Tracker",
                "No segmentation to track. Run SAM (or load masks) first.")
            return
        timeseries = self._get_full_timeseries()
        if timeseries is None:
            QMessageBox.information(
                self.window, "Run Tracker", "No image data loaded.")
            return
        self._run_tracker_and_apply(timeseries,
                                     self.window.seg_data.masks.astype(np.int32))

    def _get_full_timeseries(self):
        """Return the full (T, H, W) uint8 stack from the current source."""
        vd = self.window.video_data
        if vd is None:
            return None
        frames = getattr(vd, 'frames', None)
        if frames is not None:
            return np.asarray(frames)
        return np.stack([vd.get_frame(i) for i in range(vd.num_frames)])

    def _run_tracker_and_apply(self, timeseries, masks):
        """Run the active tracker, push undoable remap onto the stack."""
        from PyQt6.QtWidgets import QApplication
        self.window.lbl_tracker_status.setText(
            f"Running {self.tracker_service.name}…")
        QApplication.processEvents()
        try:
            remap = self.tracker_service.run(timeseries, masks)
        except Exception as e:
            log_error('controller.tracking', 'tracker raised', exc=e)
            QMessageBox.critical(
                self.window, "Tracker Error",
                f"Tracking failed:\n\n{type(e).__name__}: {e}\n\n"
                f"Run with --debug for a full traceback.")
            self.window.lbl_tracker_status.setText("")
            return
        if not remap:
            self.window.lbl_tracker_status.setText("Tracker returned no links.")
            return
        n_tracks, n_changed = self._apply_track_remap(remap)
        T = self.window.video_data.num_frames if self.window.video_data else 0
        self.window.lbl_tracker_status.setText(
            f"{self.tracker_service.name} · {n_tracks} tracks across {T} frames "
            f"· {n_changed} annotations rewritten · Cmd+Z to undo")

    def _apply_track_remap(self, remap):
        """Apply {(frame, orig_id): track_id} remap to seg + annotations.

        Each unique track_id is allocated a fresh instance_id from the seg
        allocator. Annotations matching (frame, orig_id) pairs in the
        remap get the new instance_id AND the representative name picked
        from the first annotation in each track — so the same cell across
        frames now shares name + color.

        Returns (n_unique_tracks, n_annotations_rewritten).
        """
        seg = self.window.seg_data
        if seg is None or not remap:
            return 0, 0

        # ---- Snapshot BEFORE for undo ----------------------------------
        before = {
            'masks': seg.masks.copy(),
            'colors': dict(seg.instance_colors),
            'annos': [(a, a.instance_id, a.name, a.color) for a in self.annotations],
        }

        # ---- Allocate a fresh instance_id per track --------------------
        unique_tracks = sorted({tid for tid in remap.values()})
        track_to_new_iid = {}
        for tid in unique_tracks:
            new_iid = seg.next_instance_id()
            seg.register_instance_color(new_iid)
            track_to_new_iid[tid] = new_iid

        # ---- Pick the representative name for each track ---------------
        # First (frame, orig_id) pair for a given track determines its name.
        track_to_name = {}
        for tid in unique_tracks:
            pair = next(((t, o) for (t, o), v in remap.items() if v == tid), None)
            if pair is None:
                continue
            f, o = pair
            anno = next((a for a in self.annotations
                         if a.frame_idx == f and a.instance_id == o
                         and a.class_type == 'cell'), None)
            if anno is not None:
                track_to_name[tid] = anno.name
            else:
                _, name = self._next_available_name('Cell')
                track_to_name[tid] = name

        # ---- Rewrite seg.masks -----------------------------------------
        # Process per frame so we don't accidentally double-rewrite when
        # multiple (frame, oid) pairs on the same frame share a track.
        for (f, oid), tid in remap.items():
            new_iid = track_to_new_iid[tid]
            seg.masks[f][seg.masks[f] == oid] = new_iid

        # ---- Rewrite annotation instance_id + name + color -------------
        # Index by (frame, orig_id) for O(N) total rather than O(N*remap).
        remap_by_pair = remap  # alias; same shape
        n_changed = 0
        for anno in self.annotations:
            if anno.instance_id is None:
                continue
            pair = (anno.frame_idx, int(anno.instance_id))
            tid = remap_by_pair.get(pair)
            if tid is None:
                continue
            new_iid = track_to_new_iid[tid]
            anno.instance_id = new_iid
            anno.name = track_to_name[tid]
            # Sync bbox / list color to the new track's registered color so
            # the rendered rectangle matches the seg overlay shade for the
            # cell across frames.
            anno.color = seg.instance_colors.get(new_iid, anno.color)
            n_changed += 1

        # ---- Snapshot AFTER for redo + push undoable cmd ---------------
        after = {
            'masks': seg.masks.copy(),
            'colors': dict(seg.instance_colors),
            'annos': [(a, a.instance_id, a.name, a.color) for a in self.annotations],
        }
        self._undo_stack.push(TrackingCmd(self, before, after))

        # Refresh UI.
        for anno in self.annotations:
            anno.update_visuals()
        self._show_frame_annotations(self.window._current_frame_idx)
        self.window._update_seg_overlay()
        self.state.annotations_changed.emit()
        return len(unique_tracks), n_changed

    # ------------------------------------------------------------------
    def _refresh_sam_status(self):
        """Update the status line under the SAM model selector."""
        if not SamService.available():
            text = "model: micro_sam not installed"
        elif self.sam_service.is_loaded():
            ckpt = (os.path.basename(os.path.dirname(self.sam_service.checkpoint_path))
                    if self.sam_service.checkpoint_path else "(registry)")
            text = f"model: {self.sam_service.model_type} · {ckpt} · LOADED"
        else:
            ckpt = (os.path.basename(os.path.dirname(self.sam_service.checkpoint_path))
                    if self.sam_service.checkpoint_path else "(registry)")
            ckpt_ok = (self.sam_service.checkpoint_path is None
                       or os.path.exists(self.sam_service.checkpoint_path))
            badge = "ready" if ckpt_ok else "checkpoint missing"
            text = f"model: {self.sam_service.model_type} · {ckpt} · {badge}"
        self.window.lbl_sam_status.setText(text)

    def clear_seg_mask_for_selected(self):
        """Wipe the selected cell's painted pixels on the current frame.

        Leaves all other annotations' pixels untouched. Bbox geometry
        is preserved. Pushed onto the undo stack so Cmd+Z restores the
        prior mask.
        """
        anno = self.active_annotation
        if anno is None or anno not in self.annotations:
            QMessageBox.information(
                self.window, "Clear Mask",
                "Select a cell first. Clear Mask wipes the painted pixels "
                "of the selected annotation on the current frame only.")
            return
        if anno.instance_id is None:
            QMessageBox.information(
                self.window, "Clear Mask",
                "Selected annotation has no segmentation instance.")
            return
        if anno.is_locked:
            QMessageBox.information(
                self.window, "Clear Mask",
                f"'{anno.name}' is locked. Unlock it (U) first.")
            return
        seg = self.window.seg_data
        if seg is None:
            self.window.lbl_sam_status.setText("Clear Mask: no segmentation data.")
            return

        fi = anno.frame_idx
        if fi != self.window._current_frame_idx:
            # Navigate to the cell's frame so the result is visible.
            self.window.slider_timeline.setValue(fi)

        ct = anno.class_type
        layer = seg.get_layer(ct)
        before_frame = layer[fi].copy()
        before_geom = ToolController._snap_geometry(anno)

        match = (layer[fi] == anno.instance_id)
        n_cleared = int(match.sum())
        if n_cleared == 0:
            self.window.lbl_sam_status.setText(
                f"'{anno.name}' has no painted pixels on frame {fi} — "
                f"nothing to clear.")
            return

        layer[fi][match] = 0
        after_frame = layer[fi].copy()
        self._undo_stack.push(
            SamBoxPromptCmd(self, fi, int(anno.instance_id),
                            before_frame, after_frame,
                            before_geom=before_geom,
                            after_geom=before_geom,
                            class_type=ct))  # bbox unchanged

        anno.update_visuals()
        if fi == self.window._current_frame_idx:
            self.window._update_seg_overlay()
            self._show_frame_annotations(fi)
        self.state.annotations_changed.emit()
        self.window.lbl_sam_status.setText(
            f"Cleared {n_cleared} px from {anno.name} on frame {fi} · "
            f"Cmd+Z to undo")
        log('controller.sam', 'clear seg mask',
            anno=anno.name, frame_idx=fi, pixels_cleared=n_cleared)

    def run_sam_box_prompt(self):
        """Run SAM with the selected cell's bbox as a prompt.

        Behavior is safe by construction:
          * Only paints pixels where the current seg is 0 or already
            belongs to this cell's instance_id.
          * Never overwrites another annotation's pixels.
          * Refuses on locked cells, paint-only classes (no bbox),
            off-image bboxes, etc.
          * Undoable via Cmd+Z.
        """
        log('controller.sam', 'run_sam_box_prompt: entered')

        # ---- Hard guards (clear errors, no mask changes) --------------
        if not SamService.available():
            QMessageBox.critical(self.window, "Error",
                                 "MicroSAM is not installed.")
            return
        if self.window.video_data is None:
            QMessageBox.information(self.window, "SAM Box",
                                    "Load an image first.")
            return
        anno = self.active_annotation
        if anno is None or anno not in self.annotations:
            QMessageBox.information(
                self.window, "SAM Box",
                "Select a cell first (click one in the viewer or the list).\n"
                "SAM Box uses the selected cell's bbox as the prompt.")
            return
        if anno.is_paint_only:
            QMessageBox.information(
                self.window, "SAM Box",
                f"'{anno.name}' is a {anno.class_type}, which has no bbox.\n"
                f"SAM Box only works on cells.")
            return
        if anno.instance_id is None:
            QMessageBox.information(
                self.window, "SAM Box",
                "Selected cell has no segmentation instance — add a fresh "
                "cell with A.")
            return
        if anno.is_locked:
            QMessageBox.information(
                self.window, "SAM Box",
                f"'{anno.name}' is locked. Unlock it (U) before running SAM "
                f"on it.")
            return

        # If selected cell is on another frame, jump to it so the user
        # sees the result land.
        if anno.frame_idx != self.window._current_frame_idx:
            self.window.slider_timeline.setValue(anno.frame_idx)

        # ---- Build bbox in image coords (XYXY), clamp to image bounds --
        H = self.window.video_data.height
        W = self.window.video_data.width
        x, y = anno.roi.pos()
        rw, rh = anno.roi.size()
        x0 = max(0.0, float(x))
        y0 = max(0.0, float(y))
        x1 = min(float(W), float(x) + float(rw))
        y1 = min(float(H), float(y) + float(rh))
        if x1 - x0 < 2 or y1 - y0 < 2:
            QMessageBox.information(
                self.window, "SAM Box",
                "The selected cell's bbox is empty or off-image.")
            return
        box = (x0, y0, x1, y1)

        # ---- Lazy model load (reuses sam_service caches) --------------
        try:
            self.sam_service.load()
        except FileNotFoundError as e:
            log_error('controller.sam', 'box prompt: checkpoint missing', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"{e}\n\nPlace the fine-tuned weights at "
                f"models/checkpoints/sam_hela/best.pt or switch the SAM "
                f"model in the SAM panel.")
            return
        except OSError as e:
            if getattr(e, 'errno', None) == 28:
                cache = os.path.expanduser('~/Library/Caches/micro_sam/models')
                log_error('controller.sam', 'box prompt: disk full', exc=e)
                QMessageBox.critical(
                    self.window, "Disk Full",
                    f"Out of disk space while downloading the SAM model.\n\n"
                    f"Free some space, or pick a smaller model in the SAM "
                    f"section.\n\nCache: {cache}")
                return
            log_error('controller.sam', 'box prompt: OSError', exc=e)
            QMessageBox.critical(self.window, "Error",
                                 f"OSError loading SAM model:\n\n{e}")
            return
        except Exception as e:
            log_error('controller.sam', 'box prompt: load failed', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"Failed to load SAM model:\n\n{type(e).__name__}: {e}\n\n"
                f"Run with --debug for a full traceback.")
            return

        # ---- Run SAM with the bbox prompt -----------------------------
        # Predictor isn't thread-safe; wait for any background precompute
        # to finish before we use it in the main thread.
        self._stop_embed_worker(timeout_ms=5000)
        frame = self.window.video_data.get_frame(anno.frame_idx)
        multimask = self.window.chk_sam_box_multimask.isChecked()
        log('controller.sam', 'box prompt: running',
            anno=anno.name, instance_id=anno.instance_id, box=box,
            multimask=multimask)
        try:
            # Pass image_path + frame_idx so the embedding cache kicks in;
            # subsequent prompts on the same frame are essentially free.
            mask = self.sam_service.segment_from_box(
                frame, box,
                image_path=self.window._current_file,
                frame_idx=anno.frame_idx,
                multimask_output=multimask)
        except Exception as e:
            log_error('controller.sam', 'box prompt: segment_from_box raised', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"SAM box prompt failed:\n\n{type(e).__name__}: {e}\n\n"
                f"Run with --debug for a full traceback.")
            return
        if mask is None or not bool(mask.any()):
            self.window.lbl_sam_status.setText(
                "SAM Box returned an empty mask — nothing to paint.")
            return

        # ---- Apply safely: replace own pixels with SAM's new result ---
        # Clearing the cell's existing pixels first means re-running SAM
        # on the same bbox actually shows the new prediction (otherwise
        # additive paint sees nothing new and the user sees no change).
        # Other cells' pixels are NEVER touched.
        seg = self._ensure_seg_data()
        if seg is None:
            return
        fi = anno.frame_idx
        target_id = int(anno.instance_id)
        # SAM Box is a cell-only path (vessels and capillaries have no
        # bbox to prompt with) — `seg.masks` is the cell-layer alias,
        # which is what we want here. Don't generalize this loop to
        # `seg.get_layer(...)` without first reworking the prompt UI.
        before_frame = seg.masks[fi].copy()
        before_geom = ToolController._snap_geometry(anno)

        current = seg.masks[fi]
        # Step 1: wipe this cell's prior pixels on this frame so a fresh
        # SAM run replaces them. (Pixels elsewhere on the stack are
        # unaffected — this is per-frame.)
        current[current == target_id] = 0
        # Step 2: paint SAM's mask only where the seg is empty (other
        # cells stay put).
        background = (current == 0)
        paint = mask & background
        n_painted = int(paint.sum())
        n_total_sam = int(mask.sum())
        n_blocked = n_total_sam - n_painted

        if n_painted == 0:
            # Restore the wiped state — nothing to commit.
            seg.masks[fi][:] = before_frame
            self.window.lbl_sam_status.setText(
                "SAM Box: every predicted pixel was already taken by "
                "another annotation — nothing painted.")
            return

        current[paint] = target_id

        # Step 3: tighten the bbox to fit the painted mask so the user
        # always sees an outline that matches what got segmented (the
        # SAM mask sometimes exceeds the user's original bbox).
        self._fit_bbox_to_seg(anno)
        after_geom = ToolController._snap_geometry(anno)

        after_frame = seg.masks[fi].copy()
        self._undo_stack.push(
            SamBoxPromptCmd(self, fi, target_id, before_frame, after_frame,
                            before_geom=before_geom, after_geom=after_geom))

        anno.update_visuals()
        self._show_frame_annotations(self.window._current_frame_idx)
        self.window._update_seg_overlay()
        self.state.annotations_changed.emit()
        self.window.lbl_sam_status.setText(
            f"SAM Box -> {anno.name}: painted {n_painted} px"
            + (f" ({n_blocked} blocked by other cells)" if n_blocked else "")
            + " · bbox fit to mask · Cmd+Z to undo")
        log('controller.sam', 'box prompt: done',
            anno=anno.name, painted=n_painted, blocked_by_other=n_blocked)

    def run_sam_segmentation(self):
        """Top-level Auto-segment handler.

        Additive: existing annotations and seg pixels are preserved. New
        SAM-found instances get fresh instance_ids that don't collide with
        anything already in the seg layer, and Cell_N names via the same
        gap-fill helper used for manual cells.

        Scope: just the current frame (default) OR every frame in the
        stack when the 'All frames' checkbox is on. Multi-frame mode
        updates the status line between frames; the UI may briefly hitch
        because inference is still synchronous (async lands in Phase 4.2).
        """
        log('controller.sam', 'run_sam_segmentation: entered')
        if not SamService.available():
            QMessageBox.critical(self.window, "Error", "MicroSAM is not installed.")
            return
        if self.window.video_data is None:
            QMessageBox.critical(self.window, "Error", "Load a video before running SAM.")
            return

        # Lazy model load with friendly error surfaces.
        log('controller.sam', 'requesting model load',
            model_type=self.sam_service.model_type,
            ckpt=self.sam_service.checkpoint_path)
        try:
            self.sam_service.load()
        except FileNotFoundError as e:
            log_error('controller.sam', 'checkpoint missing', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"{e}\n\nPlace the fine-tuned weights at "
                f"models/checkpoints/sam_hela/best.pt or switch the SAM "
                f"model in the SAM panel.")
            return
        except OSError as e:
            if getattr(e, 'errno', None) == 28:  # ENOSPC
                cache = os.path.expanduser('~/Library/Caches/micro_sam/models')
                log_error('controller.sam', 'disk full during model fetch', exc=e)
                QMessageBox.critical(
                    self.window, "Disk Full",
                    f"Out of disk space while downloading the SAM model.\n\n"
                    f"Free some space and try again. Any partial download "
                    f"is in:\n{cache}\n\n"
                    f"Tip: vit_l is 1.25 GB. The default sam_hela is already "
                    f"on disk; vit_b_lm is ~375 MB.")
                return
            log_error('controller.sam', 'OSError during model load', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"OSError loading SAM model:\n\n{e}\n\n"
                f"Run with --debug for a full traceback.")
            return
        except Exception as e:
            log_error('controller.sam', 'model load failed', exc=e)
            QMessageBox.critical(
                self.window, "Error",
                f"Failed to load SAM model:\n\n"
                f"{type(e).__name__}: {e}\n\n"
                f"Run with --debug for a full traceback.")
            return

        all_frames = self.window.chk_sam_all_frames.isChecked()
        if all_frames:
            frame_indices = list(range(self.window.video_data.num_frames))
        else:
            frame_indices = [self.window._current_frame_idx]
        log('controller.sam', 'scope', n_frames=len(frame_indices),
            all_frames=all_frames)

        from PyQt6.QtWidgets import QApplication, QProgressDialog
        n_total = len(frame_indices)
        scope_lbl = ("the current frame" if not all_frames
                     else f"all {n_total} frames")
        dlg = QProgressDialog(
            f"Running SAM auto-segmentation on {scope_lbl}…\n"
            "The rest of the UI is paused until this finishes.",
            "Cancel", 0, max(1, n_total), self.window)
        dlg.setWindowTitle("SAM model running")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.setAutoReset(False)
        dlg.setAutoClose(False)
        dlg.show()
        QApplication.processEvents()

        total_created = 0
        cancelled = False
        try:
            for i, fi in enumerate(frame_indices, start=1):
                if dlg.wasCanceled():
                    cancelled = True
                    break
                try:
                    n_created = self._run_sam_on_frame(fi)
                except Exception as e:
                    log_error('controller.sam', f'frame {fi} failed', exc=e)
                    dlg.close()
                    QMessageBox.critical(
                        self.window, "Error",
                        f"Segmentation failed on frame {fi}:\n\n"
                        f"{type(e).__name__}: {e}\n\n"
                        f"Created {total_created} cells before the failure.\n"
                        f"Run with --debug for a full traceback.")
                    self._refresh_sam_status()
                    return
                total_created += n_created
                dlg.setValue(i)
                dlg.setLabelText(
                    f"Frame {i}/{n_total}\n"
                    f"{total_created} cells found so far\n"
                    "The rest of the UI is paused until this finishes.")
                self.window.lbl_sam_status.setText(
                    f"SAM: frame {i}/{n_total} done · "
                    f"{total_created} cells so far")
                QApplication.processEvents()
        finally:
            dlg.close()

        if cancelled:
            self._refresh_sam_status()
            QMessageBox.information(
                self.window, "Cancelled",
                f"Auto-segmentation cancelled after {total_created} cells.")
            return

        # If All-frames + Auto-link toggle are both on, run the active
        # tracker so cells across frames collapse into single identities.
        autolink_msg = ""
        if (all_frames and self.window.chk_sam_auto_link.isChecked()
                and total_created > 0):
            timeseries = self._get_full_timeseries()
            if timeseries is not None:
                pre = len({a.instance_id for a in self.annotations
                           if a.instance_id is not None})
                self._run_tracker_and_apply(
                    timeseries,
                    self.window.seg_data.masks.astype(np.int32))
                post = len({a.instance_id for a in self.annotations
                            if a.instance_id is not None})
                autolink_msg = (f"\n\nAuto-linked: {pre} per-frame instances "
                                f"-> {post} tracks across frames. Cmd+Z to undo.")

        self._refresh_sam_status()
        scope_text = (f"all {len(frame_indices)} frames"
                      if all_frames else
                      f"frame {frame_indices[0]}")
        QMessageBox.information(
            self.window, "SAM Auto-segment Complete",
            f"Created {total_created} annotation(s) across {scope_text}.{autolink_msg}")

    # ---- Duplicate-prevention helpers (used by SAM auto-segment) ----
    _DUPLICATE_IOU_THRESHOLD = 0.30

    def _existing_mask_for(self, anno, frame_shape):
        """Return a boolean (H, W) mask representing an existing annotation
        on its frame.

        For annotations whose seg pixels are painted, that's the actual
        mask. For bbox-only annotations (manual cells with no painting),
        synthesize a bbox-shaped mask so we can still compute IoU with
        a SAM detection.
        """
        seg = self.window.seg_data
        if seg is None or anno.instance_id is None:
            return None
        painted = (seg.masks[anno.frame_idx] == anno.instance_id)
        if painted.any():
            return painted, True  # (mask, has_painted_pixels)
        # Synthesize from bbox.
        try:
            x, y = anno.roi.pos()
            w, h = anno.roi.size()
        except Exception:
            return None
        H, W = frame_shape
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m, False

    def _classify_sam_detection(self, sam_mask, frame_idx,
                                threshold=_DUPLICATE_IOU_THRESHOLD):
        """Decide what to do with one SAM-detected mask.

        Returns:
          ('new',    None)     - allocate a fresh instance + Annotation2D
          ('absorb', existing) - paint SAM pixels into existing's id (no new
                                 annotation); existing's bbox is kept as-is
                                 per the locked design choice
          ('drop',   existing) - silently drop (user already labeled this
                                 region with painted pixels)
        """
        if sam_mask.sum() == 0:
            return ('drop', None)

        best_iou = 0.0
        best_anno = None
        best_has_pixels = False
        H, W = sam_mask.shape

        for anno in self.annotations:
            if anno.frame_idx != frame_idx or anno.instance_id is None:
                continue
            res = self._existing_mask_for(anno, (H, W))
            if res is None:
                continue
            existing_mask, has_pixels = res
            # Quick reject: if bboxes don't overlap, skip the IoU calc.
            inter = int((sam_mask & existing_mask).sum())
            if inter == 0:
                continue
            union = int((sam_mask | existing_mask).sum())
            iou = inter / max(1, union)
            if iou > best_iou:
                best_iou = iou
                best_anno = anno
                best_has_pixels = has_pixels

        if best_iou < threshold:
            return ('new', None)
        if best_has_pixels:
            return ('drop', best_anno)
        return ('absorb', best_anno)

    def _run_sam_on_frame(self, frame_idx):
        """Run SAM auto-segment on a single frame, additively. Returns the
        number of new Annotation2D objects created.

        Raises on inference failure — caller decides how to surface it.
        """
        frame = self.window.video_data.get_frame(frame_idx)
        if frame is None:
            return 0
        log('controller.sam', 'frame captured',
            frame_idx=frame_idx, shape=frame.shape, dtype=str(frame.dtype))

        # SAM always runs on the raw frame, never the enhanced display.
        segmentation = self.sam_service.auto_segment(frame)
        sam_seg = segmentation.astype(np.int32)
        sam_ids = np.unique(sam_seg)
        sam_ids = sam_ids[sam_ids != 0]
        log('controller.sam', 'segmentation returned',
            frame_idx=frame_idx, n_sam_ids=len(sam_ids))
        if len(sam_ids) == 0:
            return 0

        # Make sure a seg layer exists (create empty if first-ever run).
        seg = self._ensure_seg_data()
        if seg is None:
            return 0

        dedupe = self.window.chk_sam_avoid_dupes.isChecked()
        H, W = sam_seg.shape

        n_created = 0
        n_absorbed = 0
        n_dropped = 0
        is_current = (frame_idx == self.window._current_frame_idx)

        for sid in sam_ids:
            sam_mask = (sam_seg == sid)

            if dedupe:
                decision, existing = self._classify_sam_detection(sam_mask, frame_idx)
            else:
                decision, existing = ('new', None)

            if decision == 'drop':
                n_dropped += 1
                log('controller.sam', 'dropped duplicate',
                    frame_idx=frame_idx, sam_local_id=int(sid),
                    matched_anno=existing.name if existing else None)
                continue

            if decision == 'absorb':
                # Paint SAM pixels into the existing annotation's seg slot.
                # Bbox is left unchanged per the locked design choice
                # ("trust expert labelers' bbox shape").
                seg.masks[frame_idx][sam_mask & (seg.masks[frame_idx] == 0)] = existing.instance_id
                n_absorbed += 1
                log('controller.sam', 'absorbed into bbox-only annotation',
                    frame_idx=frame_idx, sam_local_id=int(sid),
                    target=existing.name, instance_id=existing.instance_id)
                # Refresh the existing annotation's visuals so the new
                # painted pixels show up in the overlay.
                if is_current:
                    existing.update_visuals()
                continue

            # decision == 'new'
            new_id = seg.next_instance_id()
            seg.register_instance_color(new_id)
            # Only paint where background is empty (manual labels are
            # already protected, but other just-created SAM cells may
            # have grabbed adjacent pixels in this loop iteration).
            background = (seg.masks[frame_idx] == 0)
            seg.masks[frame_idx][sam_mask & background] = new_id

            mask_pixels = (seg.masks[frame_idx] == new_id)
            if not mask_pixels.any():
                seg.instance_colors.pop(new_id, None)
                continue
            ys, xs = np.where(mask_pixels)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            bw, bh = x1 - x0 + 1, y1 - y0 + 1

            n, name = self._next_available_name('Cell')
            self.anno_counter = max(self.anno_counter, n)
            color = seg.instance_colors[new_id]

            anno = Annotation2D(
                name, self.window.view_frame, self,
                start_pos=(x0, y0),
                start_size=(bw, bh),
                shape_mode='rect',
                frame_idx=frame_idx,
                instance_id=new_id,
                color=color,
            )
            anno.sig_clicked.connect(self.select_annotation)
            anno.sig_updated.connect(self._on_anno_updated)
            self.annotations.append(anno)
            anno.roi.setVisible(is_current)
            n_created += 1

        if is_current:
            self._show_frame_annotations(self.window._current_frame_idx)
            self.window._update_seg_overlay()
        self.state.annotations_changed.emit()
        log('controller.sam', 'frame done',
            frame_idx=frame_idx, n_created=n_created,
            n_absorbed=n_absorbed, n_dropped=n_dropped)
        return n_created

    def _on_seg_opacity_tools_changed(self, value):
        """Tools-panel mirror -> drive the canonical View-panel slider."""
        if self.window.slider_seg_opacity.value() == value:
            return
        self.window.slider_seg_opacity.blockSignals(True)
        self.window.slider_seg_opacity.setValue(int(value))
        self.window.slider_seg_opacity.blockSignals(False)
        self._on_seg_opacity_changed(value)

    def _on_seg_opacity_changed(self, value):
        # Keep the Tools mirror in sync without re-triggering its handler.
        tools = self.window.slider_seg_opacity_tools
        if tools.value() != value:
            tools.blockSignals(True)
            tools.setValue(int(value))
            tools.blockSignals(False)
        self.window._update_seg_overlay()

    def _on_toggle_seg(self):
        vis = not self.window._seg_visible
        self.window.set_seg_visible(vis)
        text = "Hide Seg" if vis else "Show Seg"
        # Update both mirrors; the click handler that triggered this is
        # responsible for one button's checked state — block the other's
        # signals so we don't bounce back.
        for btn in (self.window.btn_toggle_seg,
                    self.window.btn_toggle_seg_tools):
            btn.blockSignals(True)
            btn.setChecked(vis)
            btn.setText(text)
            btn.blockSignals(False)

    # ------------------------------------------------------------------
    # EXPORT / IMPORT
    # ------------------------------------------------------------------

    # ---- data collection -------------------------------------------------

    def _get_anno_rows(self, class_filter=None):
        """Return a list of dicts for every annotation.

        Parameters
        ----------
        class_filter : str | None
            If 'cell' or 'vein', only include annotations of that class.
            None → all annotations.

        Each row contains:
          name, frame, class_type, instance_id,
          x0, y0, width, height  (from the ROI — or from seg mask if
                                   chk_export_seg_bbox is checked),
          locked, shape_mode,
          inside_vein             (only for cells, when chk_export_vein_flag
                                   is checked and a seg map is loaded)
        """
        include_bbox    = self.window.chk_export_bbox.isChecked()
        use_seg_bbox    = self.window.chk_export_seg_bbox.isChecked()
        add_vein_flag   = self.window.chk_export_vein_flag.isChecked()

        # Pre-build per-frame vein masks for the vein-flag computation
        vein_masks = {}   # {frame_idx: binary H×W ndarray or None}

        rows = []
        for anno in self.annotations:
            if class_filter is not None:
                if class_filter == 'non_cell':
                    if not anno.is_paint_only:
                        continue
                elif anno.class_type != class_filter:
                    continue

            x, y = anno.roi.pos()
            w, h = anno.roi.size()

            # Optionally tighten bbox from seg pixels
            if use_seg_bbox and anno.instance_id is not None:
                seg = self.window.seg_data
                if seg is not None:
                    bbox = seg.get_instance_bbox(anno.frame_idx, anno.instance_id)
                    if bbox is not None:
                        scale = self._seg_scale()
                        if scale:
                            sx, sy = scale
                            x = bbox[0] / sx
                            y = bbox[1] / sy
                            w = bbox[2] / sx
                            h = bbox[3] / sy
                        else:
                            x, y, w, h = bbox

            row = {
                "name":        anno.name,
                "frame":       anno.frame_idx,
                "class_type":  anno.class_type,
                "instance_id": anno.instance_id or 0,
                "locked":      int(anno.is_locked),
                "shape_mode":  anno.shape_mode,
            }
            if include_bbox:
                row.update({"x0": int(round(x)), "y0": int(round(y)),
                            "width": int(round(w)), "height": int(round(h))})

            # inside_vein flag for cells
            if add_vein_flag and anno.class_type == 'cell':
                row["inside_vein"] = int(
                    self._cell_inside_vein(anno, x, y, w, h, vein_masks))

            rows.append(row)
        return rows

    def _cell_inside_vein(self, anno, x, y, w, h, vein_masks_cache):
        """Return True when any part of the cell bbox overlaps a vein pixel
        on the same frame.  Result is 1 (inside) / 0 (outside / unknown)."""
        seg = self.window.seg_data
        if seg is None:
            return False
        fi = anno.frame_idx
        if fi not in vein_masks_cache:
            vein_masks_cache[fi] = self._build_vein_mask(fi)
        vm = vein_masks_cache[fi]
        if vm is None:
            return False

        scale = self._seg_scale()
        if scale:
            sx, sy = scale
            c0 = max(0, int(round(x * sx)))
            c1 = min(seg.width,  int(round((x + w) * sx)))
            r0 = max(0, int(round(y * sy)))
            r1 = min(seg.height, int(round((y + h) * sy)))
        else:
            c0 = max(0, int(round(x)))
            c1 = min(seg.width,  int(round(x + w)))
            r0 = max(0, int(round(y)))
            r1 = min(seg.height, int(round(y + h)))

        if c1 <= c0 or r1 <= r0:
            return False
        return bool(np.any(vm[r0:r1, c0:c1]))

    def _build_vein_mask(self, frame_idx):
        """Return a boolean (H, W) mask that is True wherever any vein
        instance has a painted pixel, or None if no seg map is loaded."""
        seg = self.window.seg_data
        if seg is None or frame_idx >= seg.num_frames:
            return None
        vein_instance_ids = {
            a.instance_id
            for a in self.annotations
            if a.is_paint_only and a.instance_id is not None
               and a.frame_idx == frame_idx
        }
        if not vein_instance_ids:
            return None
        mask = seg.get_mask(frame_idx)
        vm = np.zeros(mask.shape, dtype=bool)
        for iid in vein_instance_ids:
            vm |= (mask == iid)
        return vm

    # ---- file writing helpers --------------------------------------------

    @staticmethod
    def _field_order(rows):
        """Stable column order: fixed fields first, then extras alphabetically."""
        fixed = ["name", "frame", "class_type", "instance_id",
                 "x0", "y0", "width", "height", "locked", "shape_mode"]
        extras = sorted({k for r in rows for k in r if k not in fixed})
        return [f for f in fixed if any(f in r for r in rows)] + extras

    def _write_rows(self, path, rows):
        fmt = self.window.combo_export_format.currentText()
        if fmt == "JSON":
            with open(path, "w") as f:
                json.dump({"annotations": rows}, f, indent=2)
        else:
            fields = self._field_order(rows)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore',
                                   restval=0)
                w.writeheader()
                w.writerows(rows)

    def _default_export_path(self, suffix, ext):
        stem = ""
        if self.window._current_file:
            stem = os.path.splitext(os.path.basename(self.window._current_file))[0]
            d = os.path.dirname(self.window._current_file)
        else:
            d = os.getcwd()
        return os.path.join(d, f"{stem}_{suffix}{ext}")

    def _export_dialog(self, title, suffix):
        fmt = self.window.combo_export_format.currentText()
        if fmt == "JSON":
            filt, ext = "JSON Files (*.json)", ".json"
        else:
            filt, ext = "CSV Files (*.csv)", ".csv"
        default = self._default_export_path(suffix, ext)
        path, _ = QFileDialog.getSaveFileName(self.window, title, default, filt)
        return path

    # ---- public export actions ------------------------------------------

    def export_cells(self):
        if not any(a.class_type == 'cell' for a in self.annotations):
            QMessageBox.information(self.window, "Export Cells",
                                    "No cell annotations to export.")
            return
        path = self._export_dialog("Export Cells", "cells")
        if not path:
            return
        rows = self._get_anno_rows(class_filter='cell')
        self._write_rows(path, rows)
        print(f"Cells exported → {path}  ({len(rows)} rows)")

    def export_veins(self):
        # Exports all non-cell (paint-only) annotations: vessels + capillaries.
        # Method name kept for backwards-compat with internal call sites.
        if not any(a.is_paint_only for a in self.annotations):
            QMessageBox.information(self.window, "Export Vessels",
                                    "No vessel / capillary annotations to export.")
            return
        path = self._export_dialog("Export Vessels & Capillaries", "vessels")
        if not path:
            return
        rows = self._get_anno_rows(class_filter='non_cell')
        self._write_rows(path, rows)
        print(f"Vessels/capillaries exported → {path}  ({len(rows)} rows)")

    def export_all(self):
        if not self.annotations:
            QMessageBox.information(self.window, "Export All",
                                    "No annotations to export.")
            return
        path = self._export_dialog("Export All Annotations", "annotations")
        if not path:
            return
        rows = self._get_anno_rows()
        self._write_rows(path, rows)
        print(f"All annotations exported → {path}  ({len(rows)} rows)")

    # keep old name as alias so Ctrl+S still works
    def save_annotations(self):
        self.export_all()

    def export_coco_sidecar(self):
        """Write a COCO JSON describing every committed instance mask.

        Uses the current seg_data (mask source of truth) + the metadata
        derived from this session's annotations (class_type per instance).
        Round-trip import isn't part of this path — anyone consuming the
        export reads it with standard pycocotools tooling.
        """
        from core import coco_export, sidecar

        seg = self.window.seg_data
        # Refuse only when every class layer is empty — a vessel-only or
        # capillary-only session is a legitimate thing to export.
        if seg is None or not any(
                np.any(seg.get_layer(ct)) for ct in seg.CLASS_TYPES):
            QMessageBox.information(
                self.window, "Export COCO",
                "No segmentation data to export. Paint or load a mask first.")
            return

        # Default filename: {image_stem}_coco.json next to the image.
        source = self.window._current_file
        if source:
            default = os.path.join(
                os.path.dirname(source),
                os.path.splitext(os.path.basename(source))[0] + "_coco.json")
        else:
            default = os.path.join(os.getcwd(), "annotations_coco.json")

        path, _ = QFileDialog.getSaveFileName(
            self.window, "Export COCO",
            default, "COCO JSON (*.json);;All Files (*)")
        if not path:
            return

        try:
            meta = sidecar.collect_meta_from_annotations(self.annotations)
            n = coco_export.export_coco(
                source or default, seg, meta, path)
            QMessageBox.information(
                self.window, "Export COCO",
                f"Wrote {n} annotations across {seg.num_frames} frames to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"COCO export failed:\n{e}")

    def load_annotations(self):
        if not self.window.video_data:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Import Annotations", os.getcwd(),
            "Annotation Files (*.csv *.json);;All Files (*)")
        if not path:
            return
        try:
            records = self._parse_annotation_file(path)
        except Exception as e:
            QMessageBox.critical(self.window, "Import Error", str(e))
            return
        if not records:
            QMessageBox.warning(self.window, "Import", "No annotations found.")
            return

        if self.annotations:
            reply = QMessageBox.question(
                self.window, "Import Annotations",
                f"Found {len(records)} annotations.\nClear existing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._clear_all_annotations()

        for rec in records:
            self._create_annotation_from_record(rec)

        # Repair stale name/color mismatches imported from older formats.
        self._normalize_anno_names_and_colors()

        # Show only annotations for the current frame
        self._show_frame_annotations(self.window._current_frame_idx)
        print(f"Imported {len(records)} annotations from: {path}")
        self._update_stats()

    def _parse_annotation_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.json':
            with open(path) as f:
                data = json.load(f)
            if 'annotations' in data and isinstance(data['annotations'], list):
                return self._parse_json_records(data['annotations'])
            raise ValueError("Unrecognised JSON structure.")
        return self._parse_bbox_csv(path)

    @staticmethod
    def _parse_bbox_csv(path):
        records = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'name': row.get('name', ''),
                    'frame': int(float(row.get('frame', 0))),
                    'x0': float(row.get('x0', 0)),
                    'y0': float(row.get('y0', 0)),
                    'width': float(row.get('width', 40)),
                    'height': float(row.get('height', 40)),
                    'locked': int(float(row.get('locked', 0))),
                    'shape_mode': row.get('shape_mode', ''),
                    'class_type': ToolController._normalize_class_type(row.get('class_type', 'cell')),
                })
        return records

    @staticmethod
    def _parse_json_records(annos):
        records = []
        for a in annos:
            records.append({
                'name': a.get('name', ''),
                'frame': int(a.get('frame', 0)),
                'x0': float(a.get('x0', 0)),
                'y0': float(a.get('y0', 0)),
                'width': float(a.get('width', 40)),
                'height': float(a.get('height', 40)),
                'locked': int(a.get('locked', 0)),
                'shape_mode': a.get('shape_mode', ''),
                'class_type': ToolController._normalize_class_type(a.get('class_type', 'cell')),
            })
        return records

    @staticmethod
    def _normalize_class_type(raw):
        """Coerce serialized class_type values into the current schema.

        Older exports used 'vein' for what is now called 'vessel'.
        Unknown values fall back to 'cell'.
        """
        if raw == 'vein':
            return 'vessel'
        if raw in ('cell', 'vessel', 'capillary'):
            return raw
        return 'cell'

    def _create_annotation_from_record(self, rec):
        name = rec['name']
        if not name:
            self.anno_counter += 1
            name = f"Cell_{self.anno_counter}"
        else:
            for sep in ('_', ' '):
                if sep in name:
                    try:
                        num = int(name.rsplit(sep, 1)[1])
                        self.anno_counter = max(self.anno_counter, num)
                    except (ValueError, IndexError):
                        pass

        x0, y0 = rec['x0'], rec['y0']
        w, h = rec['width'], rec['height']
        shape = rec.get('shape_mode', '') or self.current_shape_mode
        if shape not in ('rect', 'ellipse'):
            shape = self.current_shape_mode

        class_type = rec.get('class_type', 'cell')

        anno = Annotation2D(
            name, self.window.view_frame, self,
            start_pos=(x0, y0),
            start_size=(w, h),
            shape_mode=shape,
            frame_idx=int(rec.get('frame', self.window._current_frame_idx)),
            class_type=class_type,
        )
        if rec.get('locked', 0):
            anno.set_locked(True)

        anno.sig_clicked.connect(self.select_annotation)
        anno.sig_updated.connect(self._on_anno_updated)
        self.annotations.append(anno)
        # Caller follows up with _show_frame_annotations which rebuilds the
        # list widget — no need to add the row directly here.

    def _clear_all_annotations(self):
        for anno in self.annotations:
            anno.delete_ui()
        self.annotations.clear()
        self.active_annotation = None
        self.anno_counter = 0
        self.window.list_annotations.clear()
        self._undo_stack.clear()
        self.window.btn_hide_locked.setChecked(False)
        self.update_inspector()

    # ------------------------------------------------------------------
    # AUTO-SAVE
    # ------------------------------------------------------------------
    def _get_autosave_path(self):
        """autosave.json lives in the out folder when one is available,
        falling back to the legacy hidden file next to the video so older
        sessions can still recover."""
        from core import project_io
        if self.window._current_file:
            out_folder = self._resolve_out_folder()
            if out_folder:
                return os.path.join(out_folder, project_io.FILE_AUTOSAVE)
            # Fallback: next to the video, hidden-prefixed (legacy).
            d = os.path.dirname(self.window._current_file)
            stem = os.path.splitext(os.path.basename(self.window._current_file))[0]
            return os.path.join(d, f"._{stem}_autosave.json")
        return os.path.join(os.getcwd(), "._autosave_annotations.json")

    def _autosave_mode(self):
        from PyQt6.QtCore import QSettings
        from core import project_io
        s = QSettings()
        return str(s.value(project_io.SETTING_AUTOSAVE_MODE,
                            project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MODE]))

    def _autosave_mask_min_sec(self):
        from PyQt6.QtCore import QSettings
        from core import project_io
        s = QSettings()
        try:
            return int(s.value(project_io.SETTING_AUTOSAVE_MASK_MIN_SEC,
                                project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]))
        except (TypeError, ValueError):
            return project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]

    def _apply_autosave_interval(self):
        """Read the autosave interval (seconds) from QSettings and
        (re)start the timer with it. Called at startup and from the
        I/O settings dialog when the user changes the value."""
        from PyQt6.QtCore import QSettings
        from core import project_io
        s = QSettings()
        try:
            sec = int(s.value(project_io.SETTING_AUTOSAVE_INTERVAL_SEC,
                               project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_INTERVAL_SEC]))
        except (TypeError, ValueError):
            sec = project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_INTERVAL_SEC]
        sec = max(5, sec)
        self._autosave_timer.start(sec * 1000)

    def _autosave(self):
        """Periodic auto-save tick. Behavior depends on mode:

        - ``off``:   no-op.
        - ``light``: annotations + meta JSON only (cheap, <10 KB).
        - ``smart``: light + per-class mask TIFs when the seg has been
                     dirtied AND the mask-min-interval has elapsed.

        Skips while a brush stroke is in flight to avoid mid-stroke
        snapshots — the next tick picks it up.
        """
        from core import project_io, sidecar
        if not self.annotations and not (
                self.window.seg_data is not None and self._seg_dirty_since_save):
            return
        if not self.window.video_data:
            return
        mode = self._autosave_mode()
        if mode == project_io.AUTOSAVE_OFF:
            return
        if getattr(self, '_is_painting', False):
            return  # mid-stroke; try again next tick

        # Always write the lightweight autosave snapshot (annotations
        # only) so the user can recover names/classes/locks on crash.
        try:
            rows = self._get_anno_rows()
            project_io.atomic_write_json(
                self._get_autosave_path(),
                {"annotations": rows, "schema": 2},
                keep_backup=False)
            self._autosave_path = self._get_autosave_path()
        except Exception as e:
            log_error('controller.autosave', f'light snapshot failed: {e}')

        if mode != project_io.AUTOSAVE_SMART:
            return

        # Smart mode: flush masks + meta + manifest when dirty and the
        # min interval has elapsed since the last mask save. Cheap when
        # the seg is clean (no I/O at all).
        import time
        seg = self.window.seg_data
        if seg is None or not self._seg_dirty_since_save:
            return
        if time.monotonic() - self._last_mask_save_ts < self._autosave_mask_min_sec():
            return

        source = self.window._current_file
        if not source:
            return
        out_folder = self._resolve_out_folder()
        try:
            project_io.ensure_dir(out_folder)
            for ct, fname in project_io.CLASS_MASK_FILES.items():
                layer = seg.get_layer(ct)
                if layer is None or not layer.any():
                    continue
                project_io.atomic_write_tif(
                    os.path.join(out_folder, fname), layer)
            sidecar.save_meta(
                sidecar.collect_meta_from_annotations(self.annotations),
                os.path.join(out_folder, project_io.FILE_META))
            project_io.write_project_manifest(
                out_folder,
                source_video_path=source,
                frame_count=int(seg.num_frames),
                frame_size=(int(seg.height), int(seg.width)),
                class_counts=self._class_counts_for_manifest(),
                extra={"autosaved": True},
            )
            self._mark_seg_clean()
            log('controller.autosave', 'smart mask flush', out=out_folder)
        except Exception as e:
            log_error('controller.autosave', f'mask flush failed: {e}')

    def cleanup_autosave(self):
        """No-op by design: autosave.json is meant to survive crashes
        AND clean exits so the resume prompt on next open can reach it.
        Kept as a hook for tests / future cleanup policies."""
        return
