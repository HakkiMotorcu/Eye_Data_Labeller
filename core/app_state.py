from PyQt6.QtCore import QObject, pyqtSignal


class AppState(QObject):
    """Event bus emitted by ToolController for cross-component coordination.

    Phases 2-4 panels (TIFF I/O, image enhancement, SAM) subscribe to these
    instead of holding a direct reference to the controller. This keeps the
    new panels decoupled and lets the controller change internals freely.
    """

    image_loaded = pyqtSignal(int)            # num_frames
    frame_changed = pyqtSignal(int)           # frame_idx
    annotations_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)    # Annotation2D or None
    seg_changed = pyqtSignal(int)             # frame_idx
