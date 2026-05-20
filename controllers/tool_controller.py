import csv
import json
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QColor, QFont, QShortcut, QKeySequence

from core.app_state import AppState

try:
    import micro_sam
    from micro_sam import util
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


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


class MoveResizeCmd:
    """Undo/redo a bbox move or resize, optionally also restoring pixel data.

    old_mask / new_mask are full frame mask copies (int32 ndarray).
    Provided only when pixels were actually modified (translate or crop).
    """
    def __init__(self, anno, old_state, new_state,
                 ctrl=None, frame_idx=None, old_mask=None, new_mask=None):
        self.anno = anno
        self.old = old_state
        self.new = new_state
        self.ctrl = ctrl
        self.frame_idx = frame_idx
        self.old_mask = old_mask
        self.new_mask = new_mask

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
        seg.masks[self.frame_idx] = mask.copy()
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
    def __init__(self, controller, frame_idx, old_mask, new_mask):
        self.ctrl = controller
        self.frame_idx = frame_idx
        self.old_mask = old_mask
        self.new_mask = new_mask

    def undo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            seg.masks[self.frame_idx] = self.old_mask.copy()
            self.ctrl.window._update_seg_overlay()

    def redo(self):
        seg = self.ctrl.window.seg_data
        if seg is not None:
            seg.masks[self.frame_idx] = self.new_mask.copy()
            self.ctrl.window._update_seg_overlay()


class PropagateMaskCmd:
    """Undo/redo a mask-propagation operation across multiple frames."""
    def __init__(self, controller, old_masks, new_masks):
        # old_masks / new_masks: dict {frame_idx: np.ndarray copy}
        self.ctrl = controller
        self.old_masks = old_masks   # {fi: mask before propagation}
        self.new_masks = new_masks   # {fi: mask after  propagation}

    def _apply(self, masks_dict):
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        for fi, mask in masks_dict.items():
            seg.masks[fi] = mask.copy()
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
        self.class_type = class_type    # 'cell' or 'vein'

        self._seg_dirty = False         # True when seg pixels were painted/erased
        self._is_syncing = False
        self.is_selected = False
        self.is_locked = False

        x, y = start_pos
        w, h = start_size

        self.roi = self._make_roi(shape_mode, [x, y], [w, h])
        self.view.addItem(self.roi)
        self._connect_roi_signals()

        if class_type == 'vein':
            # Veins have no visible bbox
            self.roi.setPen(pg.mkPen(None))
            self.roi.hoverPen = pg.mkPen(None)
            self.roi.translatable = False
            self.roi.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for h in self.roi.getHandles():
                h.setVisible(False)
            self.roi.setVisible(False)
        else:
            self.update_visuals()

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
        self.roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.update_visuals()

    def update_visuals(self):
        if self.class_type == 'vein':
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
                   and self.class_type != 'vein')

        # Detect translate vs resize (0.5 px tolerance on size)
        _EPS = 0.5
        size_changed = (abs(old['w'] - new['w']) > _EPS or
                        abs(old['h'] - new['h']) > _EPS)

        old_mask = new_mask = None

        if not size_changed:
            # ── TRANSLATE: move pixels with the box ───────────────────────
            if has_seg:
                scale = ctrl._seg_scale()
                if scale is not None:
                    sx, sy = scale
                    old_mask = seg.masks[frame].copy()
                    seg.move_instance_pixels(
                        frame, self.instance_id,
                        (old['x'] * sx, old['y'] * sy,
                         old['w'] * sx, old['h'] * sy),
                        (new['x'] * sx, new['y'] * sy,
                         new['w'] * sx, new['h'] * sy),
                    )
                    new_mask = seg.masks[frame].copy()
                    ctrl.window._update_seg_overlay()
        else:
            # ── RESIZE ────────────────────────────────────────────────────
            shrunk = (new['w'] < old['w'] - _EPS or new['h'] < old['h'] - _EPS)
            if shrunk and has_seg:
                # Smaller box → crop pixels that now fall outside
                scale = ctrl._seg_scale()
                if scale is not None:
                    sx, sy = scale
                    old_mask = seg.masks[frame].copy()
                    seg.crop_instance_to_bbox(
                        frame, self.instance_id,
                        new['x'] * sx, new['y'] * sy,
                        new['w'] * sx, new['h'] * sy,
                    )
                    new_mask = seg.masks[frame].copy()
                    ctrl.window._update_seg_overlay()
            # Bigger box → pixels unchanged, no mask snapshot needed

        ctrl._undo_stack.push(MoveResizeCmd(
            self, old, new,
            ctrl=ctrl if old_mask is not None else None,
            frame_idx=frame,
            old_mask=old_mask,
            new_mask=new_mask,
        ))

    def delete_ui(self):
        self.view.removeItem(self.roi)

    def set_visible(self, visible):
        if self.class_type == 'vein':
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
        self._sam_predictor = None      # SAM predictor, loaded on demand

        # --- Button connections ---
        self.window.btn_add.clicked.connect(self.spawn_new_annotation)
        self.window.btn_add_vein.clicked.connect(self.spawn_vein)
        self.window.btn_delete.clicked.connect(self.delete_selected)
        self.window.btn_rename.clicked.connect(self._start_rename)
        self.window.btn_fit_bbox.clicked.connect(self._manual_fit_bbox)
        self.window.list_annotations.itemChanged.connect(self._on_list_item_edited)
        self.window.btn_lock.clicked.connect(self.lock_active)
        self.window.btn_unlock.clicked.connect(self.unlock_active)
        self.window.btn_lock_all.clicked.connect(self.lock_all)
        self.window.btn_unlock_all.clicked.connect(self.unlock_all)
        self.window.btn_hide_locked.clicked.connect(self.toggle_hide_locked)
        self.window.btn_label_colors.clicked.connect(self.toggle_label_colors)
        self.window.btn_export_cells.clicked.connect(self.export_cells)
        self.window.btn_export_veins.clicked.connect(self.export_veins)
        self.window.btn_export_all.clicked.connect(self.export_all)
        self.window.btn_import.clicked.connect(self.load_annotations)
        self.window.btn_load_seg.clicked.connect(self.load_segmentation)
        self.window.btn_run_sam.clicked.connect(self.run_sam_segmentation)
        self.window.slider_seg_opacity.valueChanged.connect(self._on_seg_opacity_changed)
        self.window.btn_toggle_seg.clicked.connect(self._on_toggle_seg)
        # Seg editing connections
        self.window._seg_mode_group.idClicked.connect(self._on_seg_mode_changed)
        self.window.slider_brush_size.valueChanged.connect(self._on_brush_size_changed)
        self.window.btn_fill_bbox.clicked.connect(self.fill_bbox_cmd)
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
            QShortcut(QKeySequence("V"),          self.window, self.spawn_vein),
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
        self._autosave_timer.start(60_000)

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
        self.window.list_annotations.blockSignals(True)
        self.window.list_annotations.clear()

        visible_annos = []
        for anno in self.annotations:
            if anno.frame_idx == frame_idx:
                if anno.class_type == 'vein':
                    anno.roi.setVisible(False)  # veins never show ROI
                else:
                    anno.roi.setVisible(True)
                visible_annos.append(anno)
                from PyQt6.QtWidgets import QListWidgetItem
                item = QListWidgetItem(anno.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.window.list_annotations.addItem(item)
            else:
                anno.roi.setVisible(False)

        # Re-select the active annotation if it's on this frame
        if self.active_annotation and self.active_annotation.frame_idx == frame_idx:
            items = self.window.list_annotations.findItems(
                self.active_annotation.name, Qt.MatchFlag.MatchExactly)
            if items:
                self.window.list_annotations.setCurrentItem(items[0])
        else:
            self.active_annotation = None
            if visible_annos:
                self.active_annotation = visible_annos[0]
                self.active_annotation.is_selected = True
                self.active_annotation.update_visuals()
                items = self.window.list_annotations.findItems(
                    visible_annos[0].name, Qt.MatchFlag.MatchExactly)
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
        if anno.class_type == 'vein':
            return  # veins have no bbox to fit
        seg = self.window.seg_data
        if seg is None:
            return
        bbox = seg.get_instance_bbox(anno.frame_idx, anno.instance_id)
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
                       x * sx, y * sy, w * sx, h * sy)

    def _move_seg_pixels(self, anno, old_geom, new_geom):
        """Move seg mask pixels when a bbox is dragged or resized."""
        if anno.class_type == 'vein':
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
                                       old_bbox, new_bbox)
        elif moved:
            seg.move_instance_pixels(anno.frame_idx, anno.instance_id,
                                     old_bbox, new_bbox)
        self.window._update_seg_overlay()

    # ------------------------------------------------------------------
    # ANNOTATION CRUD
    # ------------------------------------------------------------------
    def spawn_new_annotation(self, start_pos=None):
        if isinstance(start_pos, bool):
            start_pos = None
        if not self.window.video_data:
            return

        frame_idx = self.window._current_frame_idx

        if start_pos is None:
            cx, cy = self._get_visible_center()
            w, h = self._last_size
            start_pos = (cx - w / 2, cy - h / 2)

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
            # No match — assign a new instance
            seg = self.window.seg_data
            instance_id = None
            color = None
            if seg is not None:
                instance_id = seg.next_instance_id()
                color = seg.register_instance_color(instance_id)
                self.anno_counter = max(self.anno_counter, instance_id)
            self.anno_counter += 1
            name = f"Cell_{self.anno_counter}"

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

    def spawn_vein(self):
        """Create a bbox-less 'vein' annotation, optionally propagated across frames."""
        if not self.window.video_data:
            return
        frame_idx = self.window._current_frame_idx
        total_frames = self.window.video_data.num_frames

        seg = self.window.seg_data
        instance_id = None
        color = None
        if seg is not None:
            instance_id = seg.next_instance_id()
            color = seg.register_instance_color(instance_id,
                                                  color=(120, 80, 200))
            self.anno_counter = max(self.anno_counter, instance_id)

        self.anno_counter += 1
        name = f"Vein_{self.anno_counter}"

        # --- Propagation dialog ---
        propagate_all = False
        if total_frames > 1:
            reply = QMessageBox.question(
                self.window,
                "Propagate Vein?",
                f"Add '{name}' to all {total_frames} frames?\n\n"
                "Recommended workflow:\n"
                "  1. Click Yes so the annotation exists on every frame.\n"
                "  2. Paint the vein mask on the current frame.\n"
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
                class_type='vein',
            )
            anno.sig_clicked.connect(self.select_annotation)
            anno.sig_updated.connect(self._on_anno_updated)
            self.annotations.append(anno)
            anno.roi.setVisible(False)
            created.append(anno)

        self._undo_stack.push(AddAnnotationBatchCmd(self, created))
        self._show_frame_annotations(frame_idx)

        # Select the annotation on the current frame
        current_vein = next(
            (a for a in created if a.frame_idx == frame_idx), None)
        if current_vein:
            self.select_annotation(current_vein)

        # Auto-switch to paint mode
        self._set_seg_mode('paint')

    def _get_visible_center(self):
        vb = self.window.view_frame.getView()
        r = vb.viewRange()
        cx = (r[0][0] + r[0][1]) / 2.0
        cy = (r[1][0] + r[1][1]) / 2.0
        if self.window.video_data:
            cx = max(0, min(cx, self.window.video_data.width - 1))
            cy = max(0, min(cy, self.window.video_data.height - 1))
        return cx, cy

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
        items = self.window.list_annotations.findItems(annotation.name, Qt.MatchFlag.MatchExactly)
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
            name = current.text()
            cur = self.window._current_frame_idx
            for anno in self.annotations:
                if anno.name == name and anno.frame_idx == cur:
                    self.select_annotation(anno)
                    break

    def _on_list_item_edited(self, item):
        """Called when a list item is renamed via inline editing."""
        new_name = item.text().strip()
        if not new_name:
            # Revert to old name
            if self.active_annotation:
                item.setText(self.active_annotation.name)
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
            self.window.list_annotations.editItem(item)

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
        if anno.class_type == 'vein':
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
        index = self.annotations.index(target)

        # Erase seg mask pixels for this annotation
        self._erase_seg_for_anno(target)

        target.delete_ui()
        self.annotations.remove(target)
        self.active_annotation = None

        # Rebuild list and select next annotation on this frame
        self._show_frame_annotations(self.window._current_frame_idx)
        self.window._update_seg_overlay()
        self._undo_stack.push(DeleteAnnotationCmd(self, target, index))
        self.state.annotations_changed.emit()

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

        instance_id = int(seg.masks[frame, cy, cx])
        if instance_id == 0:
            return  # clicked on background

        # Find the annotation that owns this pixel on the current frame
        for anno in self.annotations:
            if anno.frame_idx == frame and anno.instance_id == instance_id:
                self.select_annotation(anno)
                return

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------
    def _update_stats(self):
        total = len(self.annotations)
        if total == 0:
            self.window.lbl_stats.setText("No annotations")
            return
        cur = self.window._current_frame_idx
        on_frame = sum(1 for a in self.annotations if a.frame_idx == cur)
        locked = sum(1 for a in self.annotations
                     if a.frame_idx == cur and a.is_locked)
        self.window.lbl_stats.setText(
            f"Frame: {on_frame}  |  Locked: {locked}  |  All frames: {total}")

    def _refresh_list_colors(self):
        cur = self.window._current_frame_idx
        frame_annos = [a for a in self.annotations if a.frame_idx == cur]
        self.window.list_annotations.blockSignals(True)
        for i in range(self.window.list_annotations.count()):
            item = self.window.list_annotations.item(i)
            if item is None:
                continue
            name = item.text()
            anno = next((a for a in frame_annos if a.name == name), None)
            if anno is None:
                continue
            font = item.font()
            is_active = (anno == self.active_annotation)
            if is_active and anno.is_locked:
                item.setForeground(QColor('#2ecc71'))
                font.setBold(True)
            elif is_active:
                item.setForeground(QColor('#ffd700'))
                font.setBold(True)
            elif anno.is_locked:
                item.setForeground(QColor('#2ecc71'))
                font.setBold(False)
            elif anno.class_type == 'vein':
                item.setForeground(QColor('#9370db'))
                font.setBold(False)
            else:
                item.setForeground(QColor('#cccccc'))
                font.setBold(False)
            item.setFont(font)
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
                        # Snapshot mask before stroke begins
                        seg = self.ctrl.window.seg_data
                        if seg is not None:
                            fi = self.ctrl.window._current_frame_idx
                            self.ctrl._brush_mask_snapshot = seg.masks[fi].copy()
                        self.ctrl._is_painting = True
                        vb = self.ctrl.window.view_frame.getView()
                        pos = vb.mapSceneToView(event.scenePos())
                        self.ctrl._apply_brush(pos.x(), pos.y())
                        event.accept()
                        return True
                elif etype == QEvent.Type.GraphicsSceneMouseRelease:
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.ctrl._is_painting = False
                        # Push undo command for this stroke
                        seg = self.ctrl.window.seg_data
                        snap = self.ctrl._brush_mask_snapshot
                        if seg is not None and snap is not None:
                            fi = self.ctrl.window._current_frame_idx
                            new_mask = seg.masks[fi].copy()
                            if not np.array_equal(snap, new_mask):
                                self.ctrl._undo_stack.push(
                                    BrushStrokeCmd(self.ctrl, fi, snap, new_mask))
                            self.ctrl._brush_mask_snapshot = None
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

        force = self.window.btn_force_paint.isChecked()
        if self._seg_edit_mode == 'paint':
            seg.paint_circle(frame, anno.instance_id, cx, cy, r, force=force)
            anno._seg_dirty = True
        elif self._seg_edit_mode == 'erase':
            seg.erase_circle(frame, anno.instance_id, cx, cy, r)
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

        # Assign an instance ID if the annotation doesn't have one
        if anno.instance_id is None:
            new_id = seg.next_instance_id()
            anno.instance_id = new_id
            color = seg.register_instance_color(new_id)
            anno.color = color
            anno.update_visuals()

        x, y = anno.roi.pos()
        w, h = anno.roi.size()
        force = self.window.btn_force_paint.isChecked()
        seg.fill_bbox(anno.frame_idx, anno.instance_id,
                      x * sx, y * sy, w * sx, h * sy, force=force)
        anno._seg_dirty = True
        self.window._update_seg_overlay()

    def save_seg_map(self):
        """Save the modified segmentation masks as a lossless AVI."""
        seg = self.window.seg_data
        if seg is None:
            QMessageBox.warning(self.window, "Save Seg",
                                "No segmentation data loaded.")
            return
        default = os.path.splitext(
            os.path.basename(seg.filepath))[0] + "_edited.avi"
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Segmentation Masks",
            os.path.join(os.path.dirname(seg.filepath), default),
            "AVI Video (*.avi);;All Files (*)")
        if not path:
            return
        try:
            seg.save_masks(path)
            QMessageBox.information(self.window, "Save Seg",
                                    f"Saved {seg.num_frames} frames to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self.window, "Error", str(e))

    def propagate_vein_mask(self):
        """Copy the current frame's painted mask pixels for the active annotation
        to all other frames that share the same annotation (same instance_id).

        Frames that already have pixels for this instance are skipped unless the
        user explicitly confirms overwriting — so per-frame manual edits are
        always preserved by default.
        """
        anno = self.active_annotation
        seg = self.window.seg_data
        if anno is None or anno.instance_id is None:
            QMessageBox.information(
                self.window, "Propagate Mask",
                "Select an annotation with a segmentation instance first.")
            return
        if seg is None:
            QMessageBox.information(
                self.window, "Propagate Mask",
                "No segmentation data loaded.")
            return

        source_frame = self.window._current_frame_idx
        if not np.any(seg.masks[source_frame] == anno.instance_id):
            QMessageBox.information(
                self.window, "Propagate Mask",
                "No painted pixels for this annotation on the current frame.\n\n"
                "Paint the mask here first, then use Propagate Mask to copy it "
                "to all other frames.")
            return

        # Collect all frames that have this annotation
        anno_frames = sorted({a.frame_idx for a in self.annotations
                              if a.instance_id == anno.instance_id})
        target_frames = [f for f in anno_frames if f != source_frame]

        if not target_frames:
            QMessageBox.information(
                self.window, "Propagate Mask",
                "This annotation only exists on the current frame.\n\n"
                "When adding a vein, choose 'Yes' in the propagation dialog to "
                "create the annotation on all frames first.")
            return

        # Determine which targets already have pixels
        already_painted = [f for f in target_frames
                           if np.any(seg.masks[f] == anno.instance_id)]
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
        old_masks = {fi: seg.masks[fi].copy() for fi in effective_targets}
        n_updated = seg.propagate_instance_mask(
            anno.instance_id, source_frame, effective_targets, overwrite=overwrite)
        new_masks = {fi: seg.masks[fi].copy() for fi in effective_targets}

        if n_updated > 0:
            self._undo_stack.push(PropagateMaskCmd(self, old_masks, new_masks))

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

    def load_segmentation(self):
        from core.volume_data import SegmentationData
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Load Segmentation Map",
            os.getcwd(), "Video Files (*.avi *.mp4);;All Files (*)")
        if not path:
            return
        try:
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

        # Show annotations for the current frame
        cur = self.window._current_frame_idx
        self._show_frame_annotations(cur)
        self.window._update_seg_overlay()

        all_stairs = sorted(stair_color_map.keys())
        QMessageBox.information(
            self.window, "Segmentation Loaded",
            f"Created {anno_count} bbox annotations from "
            f"{len(all_stairs)} stairs across {seg.num_frames} frames.")

    def run_sam_segmentation(self):
        if not SAM_AVAILABLE:
            QMessageBox.critical(self.window, "Error", "MicroSAM is not installed.")
            return

        if self.window.video_data is None:
            QMessageBox.critical(self.window, "Error", "Load a video before running SAM.")
            return

        # Load the predictor if not already loaded
        if self._sam_predictor is None:
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                checkpoint_path = os.path.join(
                    project_root, 'models', 'checkpoints', 'sam_hela', 'best.pt')
                if not os.path.exists(checkpoint_path):
                    QMessageBox.critical(
                        self.window, "Error",
                        f"Checkpoint not found: {checkpoint_path}\n\n"
                        f"Place the fine-tuned weights at models/checkpoints/sam_hela/best.pt.")
                    return
                self._sam_predictor = util.get_sam_model(model_type="vit_b", checkpoint_path=checkpoint_path)
            except Exception as e:
                QMessageBox.critical(self.window, "Error", f"Failed to load SAM model: {str(e)}")
                return

        # Get the current frame
        frame_idx = self.window._current_frame_idx
        frame = self.window.video_data.get_frame(frame_idx)
        if frame is None:
            QMessageBox.critical(self.window, "Error", "No frame data available.")
            return

        # Convert to RGB for SAM
        frame_rgb = np.stack([frame, frame, frame], axis=-1).astype(np.uint8)

        # Run automatic segmentation
        try:
            predictor, segmenter = get_predictor_and_segmenter(
                model_type="vit_b",
                predictor=self._sam_predictor
            )
            segmentation = micro_sam.automatic_segmentation.automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=frame_rgb,
                ndim=2,  # 2d RGB
                verbose=False
            )
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"Segmentation failed: {str(e)}")
            return

        # Create SegmentationData from the result
        from core.volume_data import SegmentationData
        # Create a custom segmentation data for single frame
        seg_data = SegmentationData.__new__(SegmentationData)
        seg_data.filepath = None
        seg_data.num_frames = self.window.video_data.num_frames
        seg_data.height = self.window.video_data.height
        seg_data.width = self.window.video_data.width
        seg_data.masks = np.zeros((seg_data.num_frames, seg_data.height, seg_data.width), dtype=np.int32)
        seg_data.masks[frame_idx] = segmentation.astype(np.int32)
        seg_data.instance_colors = {}
        # Assign colors
        unique_ids = np.unique(segmentation)
        unique_ids = unique_ids[unique_ids != 0]
        seg_data._assign_colors(list(unique_ids))

        self.window.seg_data = seg_data
        self.window._seg_visible = True

        # Clear existing annotations
        self._clear_all_annotations()

        # Create annotations from the segmentation on the frame we actually segmented.
        # (Previously hardcoded to frame 0, which silently dropped all results when
        # SAM was run on any other frame.)
        bboxes = seg_data.get_all_bboxes(frame_idx)
        anno_count = 0
        for (stair_id, blob_idx), (x0, y0, w, h) in bboxes.items():
            color = seg_data.instance_colors.get(stair_id, (255, 80, 80))
            if blob_idx == 0:
                name = f"Cell_{stair_id}"
            else:
                name = f"Cell_{stair_id}_{blob_idx}"

            anno = Annotation2D(
                name, self.window.view_frame, self,
                start_pos=(x0, y0),
                start_size=(w, h),
                shape_mode='rect',
                frame_idx=frame_idx,
                instance_id=stair_id,
                color=color,
            )
            anno.sig_clicked.connect(self.select_annotation)
            anno.sig_updated.connect(self._on_anno_updated)
            self.annotations.append(anno)
            anno.roi.setVisible(True)
            anno_count += 1

        self._show_frame_annotations(frame_idx)
        self.window._update_seg_overlay()

        QMessageBox.information(
            self.window, "SAM Segmentation Complete",
            f"Created {anno_count} annotations from SAM segmentation.")

    def _on_seg_opacity_changed(self, _value):
        self.window._update_seg_overlay()

    def _on_toggle_seg(self):
        vis = not self.window._seg_visible
        self.window.set_seg_visible(vis)
        self.window.btn_toggle_seg.setText("Hide Seg" if vis else "Show Seg")

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
            if class_filter and anno.class_type != class_filter:
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
            if a.class_type == 'vein' and a.instance_id is not None
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
        if not any(a.class_type == 'vein' for a in self.annotations):
            QMessageBox.information(self.window, "Export Veins",
                                    "No vein annotations to export.")
            return
        path = self._export_dialog("Export Veins", "veins")
        if not path:
            return
        rows = self._get_anno_rows(class_filter='vein')
        self._write_rows(path, rows)
        print(f"Veins exported → {path}  ({len(rows)} rows)")

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
            })
        return records

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

        anno = Annotation2D(
            name, self.window.view_frame, self,
            start_pos=(x0, y0),
            start_size=(w, h),
            shape_mode=shape,
            frame_idx=int(rec.get('frame', self.window._current_frame_idx)),
        )
        if rec.get('locked', 0):
            anno.set_locked(True)

        self.window.list_annotations.addItem(name)
        anno.sig_clicked.connect(self.select_annotation)
        anno.sig_updated.connect(self._on_anno_updated)
        self.annotations.append(anno)
        self._refresh_list_colors()

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
        if self.window._current_file:
            d = os.path.dirname(self.window._current_file)
            stem = os.path.splitext(os.path.basename(self.window._current_file))[0]
            return os.path.join(d, f"._{stem}_autosave.json")
        return os.path.join(os.getcwd(), "._autosave_annotations.json")

    def _autosave(self):
        if not self.annotations:
            return
        if not self.window.video_data:
            return
        path = self._get_autosave_path()
        try:
            rows = self._get_anno_rows()
            with open(path, "w") as f:
                json.dump({"annotations": rows}, f)
            self._autosave_path = path
        except Exception as e:
            print(f"Auto-save failed: {e}")

    def cleanup_autosave(self):
        path = self._autosave_path or self._get_autosave_path()
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
