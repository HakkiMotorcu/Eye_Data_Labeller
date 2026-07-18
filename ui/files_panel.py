"""Files sidebar — browse stacks and run a session queue.

Top half: a filesystem browser rooted at a folder of your choosing,
filtered to the supported image/video extensions. Double-click opens a
file in place; right-click offers Open / Add to queue.

Bottom half: the session queue (ui/session_queue.py) — the same widget
that appears on the landing page; both share one persisted list.
"""

import os

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton, QSplitter,
    QTreeView, QVBoxLayout, QWidget,
)

try:  # Qt6 moved QFileSystemModel to QtGui; older PyQt6 kept QtWidgets
    from PyQt6.QtGui import QFileSystemModel
except ImportError:  # pragma: no cover
    from PyQt6.QtWidgets import QFileSystemModel

from core.frame_source import SUPPORTED_EXTS
from ui.session_queue import SessionQueueWidget

_ROOT_KEY = 'files_panel/root'


class FilesPanel(QWidget):
    """Dock content: filesystem browser (top) + session queue (bottom)."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctrl = controller

        # ---- Browser ------------------------------------------------
        self._model = QFileSystemModel(self)
        self._model.setNameFilters([f'*{e}' for e in sorted(SUPPORTED_EXTS)])
        self._model.setNameFilterDisables(False)  # hide non-matching files

        self.tree = QTreeView(self)
        self.tree.setModel(self._model)
        self.tree.setHeaderHidden(True)
        for col in (1, 2, 3):  # size / type / date — noise here
            self.tree.hideColumn(col)
        self.tree.doubleClicked.connect(self._on_tree_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)

        self.lbl_root = QLabel("No folder chosen")
        self.lbl_root.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_root.setWordWrap(False)
        btn_root = QPushButton("Choose folder…")
        btn_root.clicked.connect(self._choose_root)
        root_row = QHBoxLayout()
        root_row.addWidget(btn_root)
        root_row.addWidget(self.lbl_root, stretch=1)

        browser = QWidget(self)
        b_lay = QVBoxLayout(browser)
        b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.addLayout(root_row)
        b_lay.addWidget(self.tree, stretch=1)

        # ---- Queue --------------------------------------------------
        self.queue = SessionQueueWidget(controller, self)

        split = QSplitter(Qt.Orientation.Vertical, self)
        split.addWidget(browser)
        split.addWidget(self.queue)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(split)

        # Restore persisted state.
        root = str(QSettings().value(_ROOT_KEY, ''))
        if root and os.path.isdir(root):
            self._set_root(root)

    def refresh(self):
        """Kept as the panel's public surface — the controller calls
        panel.refresh() after opens/saves to update status glyphs."""
        self.queue.refresh()

    # ---- Browser behavior ------------------------------------------
    def _choose_root(self):
        start = str(QSettings().value(_ROOT_KEY, '')) or os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of stacks", start)
        if path:
            QSettings().setValue(_ROOT_KEY, path)
            self._set_root(path)

    def _set_root(self, path):
        self._model.setRootPath(path)
        self.tree.setRootIndex(self._model.index(path))
        self.lbl_root.setText(os.path.basename(path) or path)
        self.lbl_root.setToolTip(path)

    def _tree_path(self, index):
        if not index.isValid():
            return None
        p = self._model.filePath(index)
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
            return p
        return None

    def _on_tree_double_click(self, index):
        p = self._tree_path(index)
        if p:
            self.ctrl.open_path(p)
            self.refresh()

    def _tree_menu(self, pos):
        p = self._tree_path(self.tree.indexAt(pos))
        if p is None:
            return
        menu = QMenu(self)
        menu.addAction("Open", lambda: self.ctrl.open_path(p))
        menu.addAction("Add to queue", lambda: self.queue.add_to_queue(p))
        menu.exec(self.tree.viewport().mapToGlobal(pos))
