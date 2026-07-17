"""Undo/redo machinery for the Eye Data Labeller.

UndoStack plus one command class per undoable operation. Extracted
verbatim from tool_controller.py (pure code motion) so undoable
operations have a home of their own; commands hold a reference to
the controller (``self.ctrl``) and call back into it at undo/redo
time, so this module stays import-light.
"""

import os

import numpy as np


# ======================================================================
# UNDO / REDO  — lightweight command pattern
# ======================================================================
class UndoStack:
    """Simple undo/redo stack. Max 50 commands.

    ``on_change`` (optional callable) fires after every push / undo /
    redo. The controller hooks _mark_seg_dirty here so EVERY undoable
    operation — including undo/redo themselves, which also mutate
    project state — flips the unsaved-changes flag. Individual
    mutation sites used to be responsible for this and most forgot,
    so painted masks could be lost on quit with the UI showing
    "Saved ✓".
    """
    def __init__(self, max_size=50, on_change=None):
        self._undo = []
        self._redo = []
        self._max = max_size
        self.on_change = on_change

    def _notify(self):
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                pass

    def push(self, cmd):
        self._undo.append(cmd)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()
        self._notify()

    def undo(self):
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        self._notify()

    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.redo()
        self._undo.append(cmd)
        self._notify()

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
    """Undo/redo deleting one cell annotation.

    Also snapshots the annotation's frame in its class layer:
    delete_selected erases the cell's painted pixels before pushing
    this command, so undo must bring those pixels back too — they
    used to be unrecoverable (bbox reappeared, mask stayed gone).
    """
    def __init__(self, controller, anno, index,
                 before_frame=None, after_frame=None):
        self.ctrl = controller
        self.anno = anno
        self.index = index
        self.before_frame = before_frame  # (H, W) copy pre-erase, or None
        self.after_frame = after_frame    # (H, W) copy post-erase, or None

    def _restore_frame(self, frame_mask):
        if frame_mask is None:
            return
        seg = self.ctrl.window.seg_data
        if seg is None:
            return
        layer = seg.get_layer(self.anno.class_type)
        if layer is None:
            return
        layer[self.anno.frame_idx][:] = frame_mask
        if self.anno.frame_idx == self.ctrl.window._current_frame_idx:
            self.ctrl.window._update_seg_overlay()

    def undo(self):
        self.ctrl._raw_restore(self.anno, index=self.index)
        self._restore_frame(self.before_frame)

    def redo(self):
        self.ctrl._raw_delete(self.anno)
        self._restore_frame(self.after_frame)


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
