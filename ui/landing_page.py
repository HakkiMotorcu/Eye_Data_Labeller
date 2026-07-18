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
        self._card_ss = (
            "#landingCard{{background:#1c1c22;border:1px solid {border};"
            "border-radius:14px;}}"
            "#landingCard QLabel{{color:#d7d7dd;background:transparent;}}")
        card.setStyleSheet(self._card_ss.format(border="#2c2c33"))
        card.setMaximumWidth(560)
        self._card = card
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

        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self.ctrl.open_io_settings)
        cl.addWidget(btn_settings)

        rlbl = QLabel("Recent — double-click or Enter opens")
        rlbl.setStyleSheet("color:#70707a;font-size:11px;"
                           "letter-spacing:.04em;padding-top:6px;")
        cl.addWidget(rlbl)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(150)
        self.recent.setStyleSheet(
            "QListWidget{background:#17171c;border:1px solid #2c2c33;"
            "border-radius:6px;} QListWidget::item{padding:4px 8px;}")
        # itemActivated covers double-click AND the Enter key.
        self.recent.itemActivated.connect(self._open_recent)
        self.recent.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._recent_menu)
        cl.addWidget(self.recent)

        # Session queue card — the same widget as the Files dock's
        # bottom half (one persisted list, ui/session_queue.py). Only
        # shown when the queue has entries, so a first run stays clean.
        from ui.session_queue import SessionQueueWidget
        qcard = QFrame()
        qcard.setObjectName("landingQueueCard")
        qcard.setStyleSheet(
            "#landingQueueCard{background:#1c1c22;border:1px solid #2c2c33;"
            "border-radius:14px;}"
            "#landingQueueCard QLabel{color:#d7d7dd;background:transparent;}")
        qcard.setMaximumWidth(380)
        ql = QVBoxLayout(qcard)
        ql.setContentsMargins(24, 24, 24, 24)
        ql.setSpacing(10)
        qtitle = QLabel("Session queue")
        qtitle.setStyleSheet("font-size:16px;font-weight:600;color:#f0f0f4;")
        ql.addWidget(qtitle)
        qsub = QLabel("Your work list — Next ▶ opens the first "
                      "unfinished stack.")
        qsub.setWordWrap(True)
        qsub.setStyleSheet("color:#9a9aa4;font-size:12px;")
        ql.addWidget(qsub)
        self.queue = SessionQueueWidget(controller, self, show_title=False)
        ql.addWidget(self.queue, stretch=1)
        self._queue_card = qcard

        outer = QHBoxLayout()
        outer.setSpacing(18)
        outer.addStretch(1)
        outer.addWidget(card)
        outer.addWidget(qcard)
        outer.addStretch(1)
        root.addLayout(outer)
        root.addStretch(1)

        self.refresh_recent()

    def refresh_recent(self):
        # Queue card: refresh statuses; hidden entirely when empty.
        self.queue.refresh()
        self._queue_card.setVisible(bool(self.queue.queue_paths()))
        self.recent.clear()
        files = self.ctrl.recent_files()
        if not files:
            it = QListWidgetItem("No recent files yet — open one to begin.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(it)
            return
        # Identical basenames (same stack name from two experiment
        # folders) get their parent folder appended so they're
        # distinguishable without hovering for the tooltip.
        from collections import Counter
        names = Counter(os.path.basename(p) for p in files)
        for p in files:
            name = os.path.basename(p)
            label = name if names[name] == 1 else (
                f"{name}  —  {os.path.basename(os.path.dirname(p)) or p}")
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, p)
            it.setToolTip(p)
            self.recent.addItem(it)

    def _open(self):
        self.ctrl.open_file_dialog()

    def _open_recent(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and not self.ctrl.open_path(p):
            # Failed open (e.g. the file vanished): re-filter so a dead
            # entry doesn't stay clickable under the error dialog.
            self.refresh_recent()

    def _recent_menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        item = self.recent.itemAt(pos)
        menu = QMenu(self)
        p = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if p:
            menu.addAction("Open", lambda: self._open_recent(item))
            menu.addAction("Remove from list", lambda: (
                self.ctrl.remove_recent_file(p), self.refresh_recent()))
            menu.addSeparator()
        if self.ctrl.recent_files():
            menu.addAction("Clear list", lambda: (
                self.ctrl.clear_recent_files(), self.refresh_recent()))
        if menu.actions():
            menu.exec(self.recent.viewport().mapToGlobal(pos))

    # Drag-drop straight onto the landing card. The card border lights
    # up while a droppable file is over the window.
    def _set_drag_highlight(self, on):
        self._card.setStyleSheet(
            self._card_ss.format(border="#3b82c4" if on else "#2c2c33"))

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and any(
                os.path.splitext(u.toLocalFile())[1].lower() in SUPPORTED_EXTS
                for u in md.urls()):
            self._set_drag_highlight(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_drag_highlight(False)

    def dropEvent(self, event):
        from PyQt6.QtCore import QTimer
        self._set_drag_highlight(False)
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                event.acceptProposedAction()
                QTimer.singleShot(0, lambda path=p: self.ctrl.open_path(path))
                return
