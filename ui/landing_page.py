"""Landing page — the first thing you see, instead of a bare dialog.

Shown as the central widget when no file is open: a titled hero with
the primary Open action, a drag-and-drop target, the recent-files
list, and a shortcut to the session queue. Opening a file (here, via
drag-drop, or the queue) switches the central stack to the annotation
view.
"""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from core.frame_source import SUPPORTED_EXTS


class LandingPage(QWidget):
    """Fileless home screen. All actions route through the controller."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctrl = controller
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        card = QFrame()
        card.setObjectName("landingCard")
        # background:transparent on the labels matters: qdarktheme's
        # app-wide QWidget rule otherwise paints each label as an
        # opaque lighter slab across the card.
        card.setStyleSheet(
            "#landingCard{background:#1c1c22;border:1px solid #2c2c33;"
            "border-radius:14px;}"
            "#landingCard QLabel{color:#d7d7dd;background:transparent;}")
        card.setMaximumWidth(560)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(36, 32, 36, 32)
        cl.setSpacing(14)

        title = QLabel("Eye Data Labeller")
        title.setStyleSheet("font-size:26px;font-weight:700;color:#f0f0f4;")
        subtitle = QLabel("Annotate cells, vessels and capillaries in "
                          "retinal microscopy stacks.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#9a9aa4;font-size:13px;")
        cl.addWidget(title)
        cl.addWidget(subtitle)

        exts = ' / '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTS))
        btn_open = QPushButton("Open image / video…")
        btn_open.setToolTip(f"Open a TIFF stack or video — {exts}  (Ctrl+O)")
        btn_open.setMinimumHeight(44)
        btn_open.setStyleSheet(
            "QPushButton{background:#3b82c4;color:white;border:none;"
            "border-radius:8px;font-size:15px;font-weight:600;}"
            "QPushButton:hover{background:#4a90d0;}")
        btn_open.clicked.connect(self._open)
        cl.addWidget(btn_open)

        drop = QLabel(f"or drop a file anywhere on this window  ({exts})")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop.setStyleSheet("color:#70707a;font-size:12px;padding:2px;")
        cl.addWidget(drop)

        row = QHBoxLayout()
        btn_queue = QPushButton("Open the session queue")
        btn_queue.setToolTip("Work through a folder of stacks with "
                             "done / in-progress status")
        btn_queue.clicked.connect(self._show_queue)
        row.addWidget(btn_queue)
        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self.ctrl.open_io_settings)
        row.addWidget(btn_settings)
        cl.addLayout(row)

        rlbl = QLabel("Recent")
        rlbl.setStyleSheet("color:#70707a;font-size:11px;"
                           "text-transform:uppercase;letter-spacing:.06em;"
                           "padding-top:6px;")
        cl.addWidget(rlbl)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(150)
        self.recent.setStyleSheet(
            "QListWidget{background:#17171c;border:1px solid #2c2c33;"
            "border-radius:6px;} QListWidget::item{padding:4px 8px;}")
        self.recent.itemDoubleClicked.connect(self._open_recent)
        cl.addWidget(self.recent)

        outer = QHBoxLayout()
        outer.addStretch(1)
        outer.addWidget(card)
        outer.addStretch(1)
        root.addLayout(outer)
        root.addStretch(1)

        self.refresh_recent()

    def refresh_recent(self):
        self.recent.clear()
        files = self.ctrl.recent_files()
        if not files:
            it = QListWidgetItem("No recent files yet — open one to begin.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(it)
            return
        for p in files:
            it = QListWidgetItem(os.path.basename(p))
            it.setData(Qt.ItemDataRole.UserRole, p)
            it.setToolTip(p)
            self.recent.addItem(it)

    def _open(self):
        self.ctrl.open_file_dialog()

    def _open_recent(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.ctrl.open_path(p)

    def _show_queue(self):
        dock = getattr(self.ctrl, '_files_dock', None)
        if dock is not None:
            dock.setVisible(True)
            dock.raise_()

    # Drag-drop straight onto the landing card.
    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and any(
                os.path.splitext(u.toLocalFile())[1].lower() in SUPPORTED_EXTS
                for u in md.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        from PyQt6.QtCore import QTimer
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                event.acceptProposedAction()
                QTimer.singleShot(0, lambda path=p: self.ctrl.open_path(path))
                return
