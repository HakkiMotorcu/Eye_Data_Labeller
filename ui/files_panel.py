"""Files sidebar — browse stacks and run a session queue.

Top half: a filesystem browser rooted at a folder of your choosing,
filtered to the supported image/video extensions. Double-click opens a
file in place; right-click offers Open / Add to queue.

Bottom half: the session queue — a curated, persisted list of stacks
with a status glyph per entry, derived from each file's resolved
output folder:

    ● saved masks exist        (done, or at least well underway)
    ◐ autosave snapshot found  (started, not saved)
    ○ untouched

"Next" opens the first unfinished entry, which is what turns a folder
of 30 stacks into a next-next-next session instead of 30 launches.
"""

import os

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QSplitter, QTreeView, QVBoxLayout, QWidget,
)

try:  # Qt6 moved QFileSystemModel to QtGui; older PyQt6 kept QtWidgets
    from PyQt6.QtGui import QFileSystemModel
except ImportError:  # pragma: no cover
    from PyQt6.QtWidgets import QFileSystemModel

from core.frame_source import SUPPORTED_EXTS
from core.debug import log, log_error

_ROOT_KEY = 'files_panel/root'
_QUEUE_KEY = 'queue/files'

_GLYPH_DONE = '●'
_GLYPH_PARTIAL = '◐'
_GLYPH_NEW = '○'


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
        self.queue_list = QListWidget(self)
        self.queue_list.itemDoubleClicked.connect(self._on_queue_double_click)
        self.queue_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(self._queue_menu)

        btn_add_current = QPushButton("+ Current")
        btn_add_current.setToolTip("Add the currently open file to the queue")
        btn_add_current.clicked.connect(self._add_current)
        btn_next = QPushButton("Next ▶")
        btn_next.setToolTip("Open the first unfinished queue entry")
        btn_next.clicked.connect(self.open_next_unfinished)
        btn_rank = QPushButton("⇅ Rank")
        btn_rank.setToolTip(
            "OPTIONAL: run the SAM model on sample frames of every queued\n"
            "stack and sort the queue by how much the model disagrees with\n"
            "your saved masks — most disagreement first, so annotation\n"
            "effort goes where the model is weakest. Slow (a few seconds\n"
            "per stack, model inference). Details: USAGE.md → 'Ranking\n"
            "the queue'.")
        btn_rank.clicked.connect(self._rank_queue)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh statuses")
        btn_refresh.clicked.connect(self.refresh)
        q_btns = QHBoxLayout()
        q_btns.addWidget(btn_add_current)
        q_btns.addWidget(btn_next)
        q_btns.addWidget(btn_rank)
        q_btns.addStretch(1)
        q_btns.addWidget(btn_refresh)

        lbl_q = QLabel("Session queue")
        lbl_q.setStyleSheet("font-weight: bold;")
        queue = QWidget(self)
        q_lay = QVBoxLayout(queue)
        q_lay.setContentsMargins(0, 6, 0, 0)
        q_lay.addWidget(lbl_q)
        q_lay.addLayout(q_btns)
        q_lay.addWidget(self.queue_list, stretch=1)

        split = QSplitter(Qt.Orientation.Vertical, self)
        split.addWidget(browser)
        split.addWidget(queue)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(split)

        # Restore persisted state.
        root = str(QSettings().value(_ROOT_KEY, ''))
        if root and os.path.isdir(root):
            self._set_root(root)
        self.refresh()

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
        menu.addAction("Add to queue", lambda: self.add_to_queue(p))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ---- Queue behavior --------------------------------------------
    def queue_paths(self):
        raw = QSettings().value(_QUEUE_KEY, []) or []
        if isinstance(raw, str):
            raw = [raw]
        return list(raw)

    def _save_queue(self, paths):
        QSettings().setValue(_QUEUE_KEY, paths)
        self.refresh()

    def add_to_queue(self, path):
        path = os.path.abspath(path)
        paths = self.queue_paths()
        if path not in paths:
            paths.append(path)
            self._save_queue(paths)
            log('files_panel', 'queued', path=path)

    def _add_current(self):
        cur = self.ctrl.window._current_file
        if cur:
            self.add_to_queue(cur)

    def _status_for(self, path):
        """(glyph, text) from the file's resolved out folder. Cheap —
        two stat calls per entry via project_io.session_summary."""
        from core import project_io
        try:
            out = self.ctrl._resolve_out_folder(path)
            summary = project_io.session_summary(out)
            if summary and summary.get('has_masks'):
                return _GLYPH_DONE, 'saved masks'
            if out and os.path.exists(os.path.join(out, 'autosave.json')):
                return _GLYPH_PARTIAL, 'autosave only'
        except Exception as e:  # never let a bad path break the panel
            log_error('files_panel', 'status check failed', exc=e, path=path)
        return _GLYPH_NEW, 'untouched'

    def refresh(self):
        self.queue_list.clear()
        current = self.ctrl.window._current_file
        for p in self.queue_paths():
            glyph, text = self._status_for(p)
            name = os.path.basename(p)
            marker = '  ← open' if (current and os.path.abspath(p) ==
                                    os.path.abspath(current)) else ''
            item = QListWidgetItem(f"{glyph}  {name}{marker}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            tip = f"{p}\n{text}"
            score = self._score_for(p)
            if score is not None:
                tip += f"\nmodel disagreement: {score:.2f}"
            item.setToolTip(tip)
            if not os.path.isfile(p):
                item.setText(f"✗  {name}  (missing)")
            self.queue_list.addItem(item)

    def _rank_queue(self):
        """Optional model-disagreement ordering — see USAGE.md."""
        paths = [p for p in self.queue_paths() if os.path.isfile(p)]
        if not paths:
            return
        scores = self.ctrl.rank_queue_by_disagreement(paths)
        if not scores:
            return
        # MERGE with stored scores — a cancelled re-rank must not wipe
        # earlier results (USAGE.md promises scored stacks keep them).
        merged = {}
        raw = QSettings().value('queue/scores', {}) or {}
        try:
            for k, v in dict(raw).items():
                merged[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
        merged.update({p: float(s) for p, s in scores.items()})
        QSettings().setValue('queue/scores', merged)
        ordered = sorted(self.queue_paths(),
                         key=lambda p: -merged.get(p, -1.0))
        self._save_queue(ordered)

    def _score_for(self, path):
        raw = QSettings().value('queue/scores', {}) or {}
        try:
            v = raw.get(path)
            return float(v) if v is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    def _on_queue_double_click(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and os.path.isfile(p):
            self.ctrl.open_path(p)
            self.refresh()

    def open_next_unfinished(self):
        for p in self.queue_paths():
            if not os.path.isfile(p):
                continue
            glyph, _ = self._status_for(p)
            if glyph != _GLYPH_DONE:
                self.ctrl.open_path(p)
                self.refresh()
                return
        self.queue_list.setToolTip("Queue finished — everything has saved masks.")
        log('files_panel', 'queue complete')

    def _queue_menu(self, pos):
        item = self.queue_list.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            p = item.data(Qt.ItemDataRole.UserRole)
            menu.addAction("Open", lambda: self._on_queue_double_click(item))
            menu.addAction("Remove from queue", lambda: self._save_queue(
                [q for q in self.queue_paths() if q != p]))
            menu.addSeparator()
        menu.addAction("Remove finished entries", lambda: self._save_queue(
            [q for q in self.queue_paths()
             if self._status_for(q)[0] != _GLYPH_DONE]))
        menu.addAction("Clear queue", lambda: self._save_queue([]))
        menu.exec(self.queue_list.viewport().mapToGlobal(pos))
