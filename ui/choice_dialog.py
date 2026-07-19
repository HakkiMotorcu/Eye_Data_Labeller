"""ChoiceDialog — a multi-option prompt whose button labels never clip.

macOS renders QMessageBox as a native alert that caps button width, so
long labels ("Save & mark in progress") get truncated. This is a plain
QDialog with a vertical stack of full-width buttons — the label always
fits, on every platform, and the choices read top-to-bottom like a
menu instead of cramped side-by-side.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout,
)


_KIND_SS = {
    'primary': ("QPushButton{background:#3b82c4;color:white;border:none;"
                "border-radius:8px;padding:10px 14px;font-size:14px;"
                "font-weight:600;text-align:left;}"
                "QPushButton:hover{background:#4a90d0;}"),
    'normal': ("QPushButton{background:#2a2a31;color:#e6e6ea;"
               "border:1px solid #3a3a42;border-radius:8px;"
               "padding:10px 14px;font-size:14px;text-align:left;}"
               "QPushButton:hover{background:#34343c;}"),
    'danger': ("QPushButton{background:transparent;color:#e2726e;"
               "border:1px solid #5a3a3a;border-radius:8px;"
               "padding:10px 14px;font-size:14px;text-align:left;}"
               "QPushButton:hover{background:#3a2a2a;}"),
}


class ChoiceDialog(QDialog):
    """Vertical, full-width option buttons. ask() returns the chosen key
    (or None if dismissed)."""

    def __init__(self, parent, title, message, options, default_key=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._result_key = None

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "font-size:16px;font-weight:600;color:#f0f0f4;")
        lbl_title.setWordWrap(True)
        root.addWidget(lbl_title)

        if message:
            lbl_msg = QLabel(message)
            lbl_msg.setWordWrap(True)
            lbl_msg.setStyleSheet("color:#9a9aa4;font-size:13px;")
            root.addWidget(lbl_msg)

        root.addSpacing(4)
        for key, label, kind in options:
            # These are descriptive labels, not menu items — escape "&"
            # so it renders literally instead of becoming a mnemonic.
            btn = QPushButton(label.replace('&', '&&'))
            btn.setStyleSheet(_KIND_SS.get(kind, _KIND_SS['normal']))
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda _c=False, k=key: self._choose(k))
            if key == default_key:
                btn.setDefault(True)
                btn.setFocus()
            root.addWidget(btn)

        self.setMinimumWidth(380)

    def _choose(self, key):
        self._result_key = key
        self.accept()

    @staticmethod
    def ask(parent, title, message, options, default_key=None):
        """options: list of (key, label, kind). kind ∈
        'primary'|'normal'|'danger'. Returns the chosen key or None."""
        dlg = ChoiceDialog(parent, title, message, options, default_key)
        dlg.exec()
        return dlg._result_key
