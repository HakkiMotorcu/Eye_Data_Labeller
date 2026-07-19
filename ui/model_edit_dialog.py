"""ModelEditDialog — add or edit one registry model (tag + base + path).

A small form dialog. Uniqueness/existence validation lives in
core.model_registry; this just collects the fields.
"""

import os

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from core import model_registry


class ModelEditDialog(QDialog):
    def __init__(self, parent=None, entry=None):
        super().__init__(parent)
        self.setModal(True)
        editing = entry is not None
        self.setWindowTitle("Edit model" if editing else "Add model")
        self.resize(460, 200)

        root = QVBoxLayout(self)
        intro = QLabel("Register a SAM checkpoint under a short tag. "
                       "Tags and file paths must each be unique.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9a9aa4;font-size:12px;")
        root.addWidget(intro)

        form = QFormLayout()
        self.ed_tag = QLineEdit((entry or {}).get('tag', ''))
        self.ed_tag.setPlaceholderText("e.g. sam_hela_v2")
        form.addRow("Tag:", self.ed_tag)

        self.combo_base = QComboBox()
        for b in model_registry.SAM_BASES:
            self.combo_base.addItem(b)
        cur_base = (entry or {}).get('base', 'vit_b')
        i = self.combo_base.findText(cur_base)
        self.combo_base.setCurrentIndex(i if i >= 0 else 0)
        self.combo_base.setToolTip(
            "The SAM architecture this checkpoint was fine-tuned from — "
            "micro_sam needs it to load the weights.")
        form.addRow("Base architecture:", self.combo_base)

        self.ed_path = QLineEdit((entry or {}).get('path', ''))
        self.ed_path.setPlaceholderText("/path/to/best.pt")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        prow = QHBoxLayout()
        prow.addWidget(self.ed_path, stretch=1)
        prow.addWidget(btn_browse)
        form.addRow("Checkpoint:", prow)
        root.addLayout(form)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _browse(self):
        start = os.path.dirname(self.ed_path.text()) or os.path.expanduser('~')
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", start,
            "Checkpoint (*.pt *.pth);;All Files (*)")
        if path:
            self.ed_path.setText(path)
            if not self.ed_tag.text().strip():
                # Seed the tag from the parent folder or filename.
                stem = os.path.basename(os.path.dirname(path)) \
                    or os.path.splitext(os.path.basename(path))[0]
                self.ed_tag.setText(stem)

    def _values(self):
        return {'tag': self.ed_tag.text().strip(),
                'path': self.ed_path.text().strip(),
                'base': self.combo_base.currentText()}

    @staticmethod
    def get_new(parent):
        dlg = ModelEditDialog(parent)
        return dlg._values() if dlg.exec() else None

    @staticmethod
    def get_edit(parent, entry):
        dlg = ModelEditDialog(parent, entry=entry)
        return dlg._values() if dlg.exec() else None
