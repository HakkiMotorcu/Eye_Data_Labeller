"""Session queue widget — one work list, two homes.

The queue itself (persisted in QSettings under ``queue/files``) is a
curated list of stacks with a status glyph per entry, derived from each
file's resolved output folder:

    ● saved masks exist        (done, or at least well underway)
    ◐ autosave snapshot found  (started, not saved)
    ○ untouched

"Next" opens the first unfinished entry, which is what turns a folder
of 30 stacks into a next-next-next session instead of 30 launches.

This widget is embedded twice: in the Files dock (annotation view) and
on the landing page. Both instances read/write the same QSettings key;
every mutation refreshes every live instance, so they can't drift.
"""

import os
import weakref

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
    QPushButton, QVBoxLayout, QWidget,
)

from core.debug import log, log_error

_QUEUE_KEY = 'queue/files'

_GLYPH_DONE = '●'
_GLYPH_PARTIAL = '◐'
_GLYPH_NEW = '○'

# Every live queue widget — a mutation through any of them refreshes
# all of them (dock + landing page never drift).
_instances = weakref.WeakSet()


class SessionQueueWidget(QWidget):
    """Queue list + controls. All actions route through the controller."""

    def __init__(self, controller, parent=None, show_title=True):
        super().__init__(parent)
        self.ctrl = controller
        _instances.add(self)

        self.queue_list = QListWidget(self)
        self.queue_list.itemDoubleClicked.connect(self._on_queue_double_click)
        self.queue_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(self._queue_menu)

        self.btn_add_current = QPushButton("+ Current")
        self.btn_add_current.setToolTip(
            "Add the currently open file to the queue")
        self.btn_add_current.clicked.connect(self._add_current)
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
        q_btns.addWidget(self.btn_add_current)
        q_btns.addWidget(btn_next)
        q_btns.addWidget(btn_rank)
        q_btns.addStretch(1)
        q_btns.addWidget(btn_refresh)

        # Empty-queue hint / "queue finished" notice — a visible label,
        # not a tooltip nobody hovers.
        self.lbl_notice = QLabel("")
        self.lbl_notice.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_notice.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if show_title:
            lbl_q = QLabel("Session queue")
            lbl_q.setStyleSheet("font-weight: bold;")
            lay.addWidget(lbl_q)
        lay.addLayout(q_btns)
        lay.addWidget(self.queue_list, stretch=1)
        lay.addWidget(self.lbl_notice)

        self.refresh()

    # ---- Persistence ------------------------------------------------
    def queue_paths(self):
        raw = QSettings().value(_QUEUE_KEY, []) or []
        if isinstance(raw, str):
            raw = [raw]
        return list(raw)

    def _save_queue(self, paths):
        QSettings().setValue(_QUEUE_KEY, paths)
        for inst in list(_instances):
            inst.refresh()

    def add_to_queue(self, path):
        path = os.path.abspath(path)
        paths = self.queue_paths()
        if path not in paths:
            paths.append(path)
            self._save_queue(paths)
            log('session_queue', 'queued', path=path)

    def _add_current(self):
        cur = self.ctrl.window._current_file
        if cur:
            self.add_to_queue(cur)

    # ---- Status -----------------------------------------------------
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
            log_error('session_queue', 'status check failed', exc=e, path=path)
        return _GLYPH_NEW, 'untouched'

    def refresh(self):
        self.queue_list.clear()
        current = self.ctrl.window._current_file
        # "+ Current" is meaningless with no file open (landing page).
        self.btn_add_current.setEnabled(bool(current))
        paths = self.queue_paths()
        self.lbl_notice.setText(
            "Queue is empty — right-click a file in the Files browser "
            "or use + Current while a stack is open." if not paths else "")
        for p in paths:
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

    # ---- Ranking ----------------------------------------------------
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

    # ---- Opening ----------------------------------------------------
    def _on_queue_double_click(self, item):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p and os.path.isfile(p):
            self.ctrl.open_path(p)
            self.refresh()

    def open_next_unfinished(self):
        paths = self.queue_paths()
        if not paths:
            return  # the empty-queue hint is already showing
        for p in paths:
            if not os.path.isfile(p):
                continue
            glyph, _ = self._status_for(p)
            if glyph != _GLYPH_DONE:
                self.ctrl.open_path(p)
                self.refresh()
                return
        self.lbl_notice.setText(
            "Queue finished — every entry has saved masks.")
        log('session_queue', 'queue complete')

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
