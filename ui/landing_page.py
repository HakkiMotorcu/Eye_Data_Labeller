"""Landing page — a simple recent-files hub.

Shown as the central widget when no file is open: the Open action, a
drag-and-drop target, and the recent list with work-status glyphs
(✓ complete / ● in progress, untouched files sorted to the bottom).
A single click opens a file. The gear opens Settings.
"""

import os

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from core.frame_source import SUPPORTED_EXTS


def _glyph_icon(glyph, color):
    """16px icon with a colored ✓ / ● — list rows show status at a
    glance without per-character rich text."""
    px = QPixmap(16, 16)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setPen(QColor(color))
    f = p.font()
    f.setPointSize(11)
    p.setFont(f)
    p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    return QIcon(px)


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
        btn_gear = QPushButton()
        btn_gear.setToolTip("Settings")
        btn_gear.setFixedSize(30, 30)
        btn_gear.setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#9a9aa4;"
            "font-size:17px;}QPushButton:hover{color:#d7d7dd;}")
        try:
            import qtawesome as qta
            btn_gear.setIcon(qta.icon('fa5s.cog', color='#9a9aa4'))
            btn_gear.setIconSize(QSize(17, 17))
        except Exception:
            btn_gear.setText("⚙")
        btn_gear.clicked.connect(self.ctrl.open_io_settings)
        trow = QHBoxLayout()
        trow.addWidget(title)
        trow.addStretch(1)
        trow.addWidget(btn_gear)
        cl.addLayout(trow)
        subtitle = QLabel("Annotate cells, vessels and capillaries in "
                          "retinal microscopy stacks.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#9a9aa4;font-size:13px;")
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

        # Shown only while no SAM checkpoint is configured — the model
        # is never demanded at startup, but it's one click from here.
        self.btn_model = QPushButton(
            "Add SAM model…   (segmentation assist is off until set)")
        self.btn_model.setStyleSheet(
            "QPushButton{background:transparent;border:1px dashed #3d3d46;"
            "border-radius:6px;color:#9a9aa4;font-size:12px;padding:6px;}"
            "QPushButton:hover{color:#d7d7dd;border-color:#55555f;}")
        self.btn_model.clicked.connect(self._add_model)
        cl.addWidget(self.btn_model)

        rlbl = QLabel("Recent — click to open   ·   ✓ complete   "
                      "● in progress")
        rlbl.setStyleSheet("color:#70707a;font-size:11px;"
                           "letter-spacing:.04em;padding-top:6px;")
        cl.addWidget(rlbl)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(190)
        self.recent.setStyleSheet(
            "QListWidget{background:#17171c;border:1px solid #2c2c33;"
            "border-radius:6px;} QListWidget::item{padding:4px 8px;}")
        # Single click opens (user decision: no confirm when nothing is
        # open — the leave dialog guards live sessions). Enter works too.
        self.recent.itemClicked.connect(self._open_recent)
        self.recent.itemActivated.connect(self._open_recent)
        self.recent.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._recent_menu)
        cl.addWidget(self.recent)

        outer = QHBoxLayout()
        outer.addStretch(1)
        outer.addWidget(card)
        outer.addStretch(1)
        root.addLayout(outer)
        root.addStretch(1)

        self.refresh_recent()

    def _model_missing(self):
        svc = getattr(self.ctrl, 'sam_service', None)
        ck = getattr(svc, 'checkpoint_path', None)
        return bool(ck) and not os.path.exists(ck)

    def _add_model(self):
        self.ctrl.choose_model_checkpoint()
        self.refresh_recent()

    def _status_for(self, path):
        from core import project_io
        try:
            out = self.ctrl._resolve_out_folder(path)
            return project_io.file_status(out) if out else None
        except Exception:
            return None

    def refresh_recent(self):
        self.btn_model.setVisible(self._model_missing())
        self.recent.clear()
        files = self.ctrl.recent_files()
        if not files:
            it = QListWidgetItem("No recent files yet — open one to begin.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(it)
            return
        # Worked-on files first (recency order), untouched at the
        # bottom — the top of the list is always "continue where you
        # left off".
        statuses = {p: self._status_for(p) for p in files}
        files = ([p for p in files if statuses[p] is not None]
                 + [p for p in files if statuses[p] is None])
        # Identical basenames (same stack name from two experiment
        # folders) get their parent folder appended so they're
        # distinguishable without hovering for the tooltip.
        from collections import Counter
        names = Counter(os.path.basename(p) for p in files)
        icon_done = _glyph_icon('✓', '#4caf82')
        icon_wip = _glyph_icon('●', '#e6b84c')
        for p in files:
            name = os.path.basename(p)
            label = name if names[name] == 1 else (
                f"{name}  —  {os.path.basename(os.path.dirname(p)) or p}")
            it = QListWidgetItem(label)
            status = statuses[p]
            if status == 'complete':
                it.setIcon(icon_done)
            elif status == 'in_progress':
                it.setIcon(icon_wip)
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
