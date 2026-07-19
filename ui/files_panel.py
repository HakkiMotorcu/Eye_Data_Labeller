"""Files sidebar — a real file explorer with work-status glyphs.

A filesystem tree rooted at a folder of your choosing, filtered to the
supported image/video extensions. Each supported file shows its work
status at the right edge, derived from its out folder's project.json:

    ✓  complete     (user explicitly marked it done at close/switch)
    ●  in progress  (session artifacts exist without that mark)
       untouched    (no glyph)

Double-click opens a file in place — the status-aware leave dialog
(Save & mark complete / in progress / Cancel) guards the current
session. "Next ▶" opens the first top-level file in the folder that
isn't marked complete, which is what turns a folder of 30 stacks into
a next-next-next session instead of 30 launches.
"""

import os

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
    QStyledItemDelegate, QTreeView, QVBoxLayout, QWidget,
)

try:  # Qt6 moved QFileSystemModel to QtGui; older PyQt6 kept QtWidgets
    from PyQt6.QtGui import QFileSystemModel
except ImportError:  # pragma: no cover
    from PyQt6.QtWidgets import QFileSystemModel

from core.frame_source import SUPPORTED_EXTS
from core.debug import log, log_error

_ROOT_KEY = 'files_panel/root'

_COLOR_DONE = QColor('#4caf82')
_COLOR_WIP = QColor('#e6b84c')


class _StatusDelegate(QStyledItemDelegate):
    """Paints ✓ / ● at the right edge of supported files' rows."""

    def __init__(self, panel):
        super().__init__(panel)
        self._panel = panel

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        path = self._panel._model.filePath(index)
        if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTS:
            return
        status = self._panel.status_cached(path)
        if status is None:
            return
        glyph, color = (('✓', _COLOR_DONE) if status == 'complete'
                        else ('●', _COLOR_WIP))
        painter.save()
        painter.setPen(color)
        painter.drawText(
            option.rect.adjusted(0, 0, -6, 0),
            int(Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter), glyph)
        painter.restore()


class FilesPanel(QWidget):
    """Dock content: explorer tree + Next-unfinished, nothing else."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctrl = controller
        self._status_cache = {}

        self._model = QFileSystemModel(self)
        self._model.setNameFilters([f'*{e}' for e in sorted(SUPPORTED_EXTS)])
        self._model.setNameFilterDisables(False)  # hide non-matching files

        self.tree = QTreeView(self)
        self.tree.setModel(self._model)
        self.tree.setHeaderHidden(True)
        for col in (1, 2, 3):  # size / type / date — noise here
            self.tree.hideColumn(col)
        self.tree.setItemDelegateForColumn(0, _StatusDelegate(self))
        self.tree.doubleClicked.connect(self._on_tree_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.setMinimumWidth(0)

        btn_root = QPushButton("Choose folder…")
        btn_root.clicked.connect(self._choose_root)
        self.lbl_root = QLabel("No folder chosen")
        self.lbl_root.setStyleSheet("color: #888; font-size: 11px;")
        # Ignored horizontal policy: a long path in this label must
        # never dictate the dock's minimum width (this is what made
        # the sidebar un-resizable — the full path became a floor).
        self.lbl_root.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Preferred)
        root_row = QHBoxLayout()
        root_row.addWidget(btn_root)
        root_row.addWidget(self.lbl_root, stretch=1)

        btn_next = QPushButton("Next ▶")
        btn_next.setToolTip(
            "Open the first file in this folder (top level) that isn't "
            "marked ✓ complete")
        btn_next.clicked.connect(self.open_next_unfinished)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh status glyphs")
        btn_refresh.clicked.connect(self.refresh)
        next_row = QHBoxLayout()
        next_row.addWidget(btn_next)
        next_row.addStretch(1)
        next_row.addWidget(btn_refresh)

        self.lbl_note = QLabel("")
        self.lbl_note.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_note.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(root_row)
        lay.addLayout(next_row)
        lay.addWidget(self.tree, stretch=1)
        lay.addWidget(self.lbl_note)

        root = str(QSettings().value(_ROOT_KEY, ''))
        if root and os.path.isdir(root):
            self._set_root(root)

    # ---- Public surface (controller calls this after opens/saves) ---
    def refresh(self):
        self._status_cache.clear()
        self.lbl_note.setText("")
        self.tree.viewport().update()

    def status_cached(self, path):
        """Status for a file, cached until the next refresh — the
        delegate calls this on every paint."""
        if path in self._status_cache:
            return self._status_cache[path]
        from core import project_io
        status = None
        try:
            out = self.ctrl._resolve_out_folder(path)
            status = project_io.file_status(out) if out else None
        except Exception as e:  # never let a bad path break painting
            log_error('files_panel', 'status check failed', exc=e, path=path)
        self._status_cache[path] = status
        return status

    # ---- Root selection --------------------------------------------
    def _choose_root(self):
        start = str(QSettings().value(_ROOT_KEY, '')) or os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of stacks", start)
        if path:
            QSettings().setValue(_ROOT_KEY, path)
            self._set_root(path)

    def _set_root(self, path):
        path = os.path.normpath(path)
        self._model.setRootPath(path)
        self.tree.setRootIndex(self._model.index(path))
        self.lbl_root.setText(os.path.basename(path) or path)
        self.lbl_root.setToolTip(path)

    def _root_path(self):
        p = self._model.rootPath()
        return p if p and os.path.isdir(p) else None

    # ---- Opening ----------------------------------------------------
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

    def open_next_unfinished(self):
        """First top-level supported file not marked ✓ and not already
        open. Top level only: subfolders like out/ hold mask TIFs that
        would match the extension filter but aren't stacks to open."""
        root = self._root_path()
        if root is None:
            self.lbl_note.setText("Choose a folder first.")
            return
        current = self.ctrl.window._current_file
        candidates = sorted(
            os.path.join(root, n) for n in os.listdir(root)
            if os.path.splitext(n)[1].lower() in SUPPORTED_EXTS
            and os.path.isfile(os.path.join(root, n)))
        if not candidates:
            self.lbl_note.setText("No supported files in this folder.")
            return
        for p in candidates:
            if current and os.path.abspath(p) == os.path.abspath(current):
                continue
            if self.status_cached(p) != 'complete':
                if self.ctrl.open_path(p):
                    self.refresh()
                return
        self.lbl_note.setText("All files here are marked ✓ complete.")
        log('files_panel', 'folder complete', root=root)

    # ---- Context menu ----------------------------------------------
    def _mark(self, path, status):
        from core import project_io
        try:
            out = self.ctrl._resolve_out_folder(path)
            if out:
                project_io.write_status(out, status)
        except OSError as e:
            log_error('files_panel', 'mark failed', exc=e, path=path)
        self.refresh()

    def _tree_menu(self, pos):
        from core import project_io
        p = self._tree_path(self.tree.indexAt(pos))
        if p is None:
            return
        menu = QMenu(self)
        menu.addAction("Open", lambda: (self.ctrl.open_path(p),
                                        self.refresh()))
        menu.addSeparator()
        menu.addAction("Mark ✓ complete",
                       lambda: self._mark(p, project_io.STATUS_COMPLETE))
        menu.addAction("Mark ● in progress",
                       lambda: self._mark(p, project_io.STATUS_IN_PROGRESS))
        menu.addAction("Clear status", lambda: self._mark(p, None))
        menu.exec(self.tree.viewport().mapToGlobal(pos))
