"""Annotation2D — one on-canvas annotation (pyqtgraph ROI wrapper).

Extracted verbatim from tool_controller.py (pure code motion).
Cells own a visible, draggable rect ROI; paint-only classes
(vessel / capillary) render nothing and exist as list entries over
their painted pixels.
"""

import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QColor

from controllers.commands import MoveResizeCmd


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
        # _snap_geometry is a staticmethod on the controller class;
        # resolve it through the instance so this module needs no
        # (circular) ToolController import.
        self.controller._geometry_snapshot = self.controller._snap_geometry(self)

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
        new = self.controller._snap_geometry(self)
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
