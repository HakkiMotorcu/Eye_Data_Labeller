import sys
import os

# Honor --debug BEFORE importing anything else, so the debug flag is
# already live by the time submodules read it.
if '--debug' in sys.argv:
    os.environ['EYE_LABELLER_DEBUG'] = '1'
    sys.argv.remove('--debug')

# Point Qt at PyQt6's bundled plugins BEFORE importing any Qt module.
# Why: conda-forge envs often have a qt.conf file (from a Qt5 dep
# pulled in transitively) that points the Qt prefix at the env root,
# which then sends Qt looking for plugins in the wrong place. PyQt6
# itself ships its plugins inside its own site-packages dir; that's
# the path we want. Overriding the env var here trumps qt.conf and
# trumps any QT_QPA_PLATFORM_PLUGIN_PATH the user has set globally.
try:
    import PyQt6 as _pyqt6
    _plugins = os.path.join(os.path.dirname(_pyqt6.__file__), 'Qt6', 'plugins')
    if os.path.isdir(_plugins):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
            _plugins, 'platforms')
        os.environ['QT_PLUGIN_PATH'] = _plugins
except Exception:
    # If PyQt6 isn't importable at all, the next import will produce
    # a clearer error than anything we could craft here.
    pass

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QColor, QPen
from PyQt6.QtCore import Qt, QRectF, QSettings, QTimer
from core.volume_data import VideoData
from core.frame_source import (
    TiffFrameSource, load_frame_source, VIDEO_EXTS as _VIDEO_EXTS,
    TIFF_EXTS as _TIFF_EXTS,
)
from core.debug import (
    log, is_debug, set_debug, log_startup_banner,
    install_qt_message_handler, SETTING_DEBUG_KEY,
)
from ui.main_window import MainWindow
from controllers.tool_controller import ToolController


def _selftest():
    """Headless startup check — used by CI on the built bundle.

    Reaching this function already proves the full import chain
    (PyQt6, numpy, cv2, torch, micro_sam, pyqtgraph …) survived, which
    is where every past bundle breakage manifested. On top of that,
    construct the real MainWindow + ToolController against a synthetic
    stack, offscreen, so widget wiring is exercised too.
    """
    import platform
    import tempfile
    print(f"selftest: python {sys.version.split()[0]} "
          f"on {platform.platform()} frozen={getattr(sys, 'frozen', False)}")
    try:
        # The probe loop must be INSIDE the try: in the windowed
        # (console=False) bundle an uncaught import error pops a modal
        # error dialog that a headless CI runner can never dismiss —
        # the job would hang for hours instead of failing fast.
        for mod in ('numpy', 'cv2', 'torch', 'tifffile', 'zarr',
                    'pyqtgraph', 'micro_sam', 'PyQt6.QtCore'):
            m = sys.modules.get(mod) or __import__(mod, fromlist=['__name__'])
            ver = getattr(m, '__version__', None) or getattr(
                m, 'QT_VERSION_STR', '?')
            print(f"selftest: {mod} {ver}")
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication([])
        app.setOrganizationName("EyeDataLabeller")
        app.setApplicationName("Eye Data Labeller")

        import numpy as np
        import tifffile
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'selftest.tif')
            tifffile.imwrite(
                path, (np.random.default_rng(0)
                       .integers(0, 255, (3, 64, 64))).astype('uint8'))
            data = load_frame_source(path)
            window = MainWindow(video_data=data)
            controller = ToolController(window)
            window._controller = controller
            window._current_file = path
            controller.spawn_new_annotation()
            controller.delete_selected()
            # Mark clean BEFORE close(): the closeEvent unsaved-changes
            # guard would otherwise open a modal prompt that no one can
            # answer in a headless CI run — the selftest would hang.
            controller._mark_seg_clean()
            # Tear down deterministically — letting GC destroy the
            # pyqtgraph ViewBox during interpreter shutdown prints
            # scary (but harmless) RuntimeError tracebacks in CI logs.
            window.close()
            del controller, window, data
            app.processEvents()
        print("selftest: MainWindow + ToolController constructed OK")
    except Exception:
        import traceback
        traceback.print_exc()
        print("selftest: FAILED")
        return 1
    print("selftest: PASS")
    return 0


# load_frame_source now lives in core.frame_source (imported above) so
# File>Open, drag-drop, and the session queue can use it without
# importing this entry-point module.

# Must come after imports are complete — _selftest uses load_frame_source.
if '--selftest' in sys.argv:
    sys.exit(_selftest())


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

    # Detailed-logging toggle: --debug / env var always wins; otherwise
    # honor the persisted in-app setting (I/O Settings → Debugging).
    if not is_debug():
        v = QSettings().value(SETTING_DEBUG_KEY, False)
        set_debug(str(v).lower() in ('1', 'true', 'yes', 'on'))
    install_qt_message_handler()
    log_startup_banner()

    # Dark theme — consistent palette across all widgets.
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except Exception as e:
        print(f"Dark theme unavailable ({e}); falling back to system theme.")

    icon = _make_eye_icon(64)
    app.setWindowIcon(icon)

    # An optional path on the command line opens straight into the
    # annotation view; otherwise the window opens on the landing page.
    cli_path = next((a for a in sys.argv[1:] if not a.startswith('-')), None)

    print("Launching Interface...")
    window = MainWindow(video_data=None)
    window.setWindowIcon(icon)
    controller = ToolController(window)
    window._controller = controller
    window.install_landing()
    window.show_landing()
    window._update_title()

    app.aboutToQuit.connect(controller.cleanup_autosave)

    window.show()
    # open_path (queued so its dialogs get a visible parent) handles the
    # load, resume prompt, and first-frame precompute — same path as
    # File>Open, so the landing/annotation switch is uniform.
    if cli_path:
        QTimer.singleShot(0, lambda: controller.open_path(cli_path))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
