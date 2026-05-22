import sys
import os

# Honor --debug BEFORE importing anything else, so the debug flag is
# already live by the time submodules read it.
if '--debug' in sys.argv:
    os.environ['EYE_LABELLER_DEBUG'] = '1'
    sys.argv.remove('--debug')

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QColor, QPen
from PyQt6.QtCore import Qt, QRectF
from core.volume_data import VideoData
from core.frame_source import TiffFrameSource
from core.debug import log, is_debug
from ui.main_window import MainWindow
from controllers.tool_controller import ToolController


_VIDEO_EXTS = {'.avi', '.mp4', '.mkv', '.mov'}
_TIFF_EXTS = {'.tif', '.tiff'}


def load_frame_source(path):
    """Open a file as the right kind of frame source based on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _TIFF_EXTS:
        return TiffFrameSource(path)
    if ext in _VIDEO_EXTS:
        return VideoData(path)
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Accepted: {sorted(_VIDEO_EXTS | _TIFF_EXTS)}")


def _make_eye_icon(size=64):
    """Generate a simple eye icon programmatically — no external file needed."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    w, h = size, size
    # Dark background circle
    p.setBrush(QColor(30, 30, 45))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, w, h)

    # Eye outline (white almond shape)
    margin = w * 0.08
    eye_rect = QRectF(margin, h * 0.28, w - 2 * margin, h * 0.44)
    path = QPainterPath()
    cx, cy = eye_rect.center().x(), eye_rect.center().y()
    ew, eh = eye_rect.width() / 2, eye_rect.height() / 2
    path.moveTo(cx - ew, cy)
    path.cubicTo(cx - ew, cy - eh * 1.6, cx + ew, cy - eh * 1.6, cx + ew, cy)
    path.cubicTo(cx + ew, cy + eh * 1.6, cx - ew, cy + eh * 1.6, cx - ew, cy)
    path.closeSubpath()
    p.setBrush(QColor(220, 230, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPath(path)

    # Iris
    iris_r = eh * 0.85
    p.setBrush(QColor(60, 140, 210))
    p.drawEllipse(QRectF(cx - iris_r, cy - iris_r, iris_r * 2, iris_r * 2))

    # Pupil
    pupil_r = iris_r * 0.5
    p.setBrush(QColor(15, 15, 20))
    p.drawEllipse(QRectF(cx - pupil_r, cy - pupil_r, pupil_r * 2, pupil_r * 2))

    # Specular highlight
    hl_r = pupil_r * 0.35
    p.setBrush(QColor(255, 255, 255, 200))
    p.drawEllipse(QRectF(cx + pupil_r * 0.3 - hl_r,
                         cy - pupil_r * 0.55 - hl_r, hl_r * 2, hl_r * 2))

    # Rim on the almond
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(100, 170, 230), max(1, size // 32)))
    p.drawPath(path)

    p.end()
    return QIcon(px)


def pick_video_file(parent=None):
    """Open a file dialog and return the selected path (or None if cancelled)."""
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Open Image or Video",
        os.getcwd(),
        "All supported (*.tif *.tiff *.avi *.mp4 *.mkv *.mov);;"
        "TIFF (*.tif *.tiff);;Video (*.avi *.mp4 *.mkv *.mov);;All Files (*)",
    )
    return path if path else None


def main():
    if is_debug():
        log('main', 'launching with debug enabled', argv=sys.argv)
    app = QApplication(sys.argv)
    app.setOrganizationName("EyeDataLabeller")
    app.setApplicationName("Eye Data Labeller")
    # QSettings without explicit identity ends up in a generic
    # "Unknown organization" bucket on macOS — pin it so the I/O
    # settings dialog's values are stable across launches.

    # Dark theme — consistent palette across all widgets.
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except Exception as e:
        print(f"Dark theme unavailable ({e}); falling back to system theme.")

    icon = _make_eye_icon(64)
    app.setWindowIcon(icon)

    # --- first file pick (before window exists) ---
    file_path = pick_video_file()
    if not file_path:
        sys.exit(0)

    print(f"Loading: {file_path}")
    try:
        data = load_frame_source(file_path)
    except Exception as e:
        QMessageBox.critical(None, "Load Error", str(e))
        sys.exit(1)

    print("Launching Interface...")
    window = MainWindow(video_data=data)
    window.setWindowIcon(icon)
    controller = ToolController(window)

    window._controller = controller
    window._current_file = file_path
    window._update_title()
    # Kick off background precompute of the current frame's SAM embedding
    # so the first SAM Box click is fast.
    controller.on_image_loaded()

    app.aboutToQuit.connect(controller.cleanup_autosave)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
