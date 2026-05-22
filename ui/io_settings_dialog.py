"""Modal dialog for project I/O settings.

Lets the user pick:

* Where the per-video output folder lives:
    - ``<video_dir>/out/<stem>/`` (default subfolder under an out/ sibling)
    - ``<video_dir>/<stem>_out/`` (folder right next to the video)
    - ``<custom_root>/<stem>/``   (one global "projects root", configurable)
* Auto-save mode (off / light / smart) and interval(s).

Values persist via ``QSettings`` so they survive across launches.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
)

from core import project_io


class IOSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Output & autosave settings")
        self.setModal(True)
        self._settings = QSettings()

        # ----- Read current values from QSettings ------------------
        mode = str(self._settings.value(
            project_io.SETTING_OUTPUT_MODE,
            project_io.DEFAULTS[project_io.SETTING_OUTPUT_MODE]))
        custom_root = str(self._settings.value(
            project_io.SETTING_OUTPUT_CUSTOM_ROOT,
            project_io.DEFAULTS[project_io.SETTING_OUTPUT_CUSTOM_ROOT]))
        autosave_mode = str(self._settings.value(
            project_io.SETTING_AUTOSAVE_MODE,
            project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MODE]))
        try:
            interval_sec = int(self._settings.value(
                project_io.SETTING_AUTOSAVE_INTERVAL_SEC,
                project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_INTERVAL_SEC]))
        except (TypeError, ValueError):
            interval_sec = project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_INTERVAL_SEC]
        try:
            mask_min_sec = int(self._settings.value(
                project_io.SETTING_AUTOSAVE_MASK_MIN_SEC,
                project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]))
        except (TypeError, ValueError):
            mask_min_sec = project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]

        # Model download URL — what we hit on first SAM use if the
        # checkpoint isn't on disk yet.
        from core import model_download
        model_url = str(self._settings.value(
            model_download.SETTINGS_KEY,
            model_download.DEFAULT_SAM_HELA_URL))
        # Optional explicit local checkpoint path — takes precedence
        # over the download URL when set, so collaborators can point
        # the app at a file they already have on disk.
        model_local = str(self._settings.value(
            model_download.LOCAL_PATH_SETTINGS_KEY, ""))

        # ----- Output mode group ------------------------------------
        out_lbl = QLabel("Output folder")
        out_lbl.setStyleSheet("font-weight: bold;")
        self.rb_sub  = QRadioButton("Subfolder under out/ next to the video "
                                     "( <dir>/out/<stem>/ )")
        self.rb_pref = QRadioButton("Folder next to the video "
                                     "( <dir>/<stem>_out/ )")
        self.rb_cust = QRadioButton("Custom root (one folder for every "
                                     "project)")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_sub)
        grp.addButton(self.rb_pref)
        grp.addButton(self.rb_cust)
        grp.setExclusive(True)
        {
            project_io.OUTPUT_MODE_SUBFOLDER: self.rb_sub,
            project_io.OUTPUT_MODE_PREFIXED:  self.rb_pref,
            project_io.OUTPUT_MODE_CUSTOM:    self.rb_cust,
        }.get(mode, self.rb_sub).setChecked(True)

        custom_row = QHBoxLayout()
        self.ed_custom = QLineEdit(custom_root)
        self.ed_custom.setPlaceholderText(
            "/Users/.../EyeLabellerProjects   (used when 'Custom root' is "
            "selected)")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_custom_root)
        custom_row.addWidget(self.ed_custom, stretch=1)
        custom_row.addWidget(btn_browse)

        self.lbl_preview = QLabel("Preview: —")
        self.lbl_preview.setStyleSheet("color: #888; font-family: monospace; "
                                         "font-size: 11px;")
        for rb in (self.rb_sub, self.rb_pref, self.rb_cust):
            rb.toggled.connect(self._update_preview)
        self.ed_custom.textChanged.connect(self._update_preview)

        # ----- Autosave group ---------------------------------------
        auto_lbl = QLabel("Auto-save")
        auto_lbl.setStyleSheet("font-weight: bold;")
        self.combo_autosave = QComboBox()
        self.combo_autosave.addItem("Off",   project_io.AUTOSAVE_OFF)
        self.combo_autosave.addItem("Light — annotations + meta only",
                                      project_io.AUTOSAVE_LIGHT)
        self.combo_autosave.addItem("Smart — light + masks when dirty + "
                                      "min interval elapsed",
                                      project_io.AUTOSAVE_SMART)
        idx = self.combo_autosave.findData(autosave_mode)
        self.combo_autosave.setCurrentIndex(idx if idx >= 0 else 1)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(5, 600)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.setValue(interval_sec)
        self.spin_interval.setToolTip(
            "How often the autosave timer fires. Light mode writes on "
            "every tick; Smart mode only flushes masks when dirty.")

        self.spin_mask_min = QSpinBox()
        self.spin_mask_min.setRange(30, 3600)
        self.spin_mask_min.setSuffix(" s")
        self.spin_mask_min.setValue(mask_min_sec)
        self.spin_mask_min.setToolTip(
            "Smart mode only: minimum seconds between two on-disk mask "
            "flushes, regardless of how often the seg gets dirtied.")

        # SAM-HeLa checkpoint download URL.
        self.ed_model_url = QLineEdit(model_url)
        self.ed_model_url.setPlaceholderText(
            "https://huggingface.co/…/best.pt   "
            "(downloaded on first use if not already on disk)")
        self.ed_model_url.setToolTip(
            "Where to fetch sam_hela/best.pt when the file is missing on "
            "disk. Set this to a public HTTPS URL — Hugging Face, GitHub "
            "Release asset, S3, etc.")

        # Local checkpoint file (overrides the download URL when set).
        self.ed_model_local = QLineEdit(model_local)
        self.ed_model_local.setPlaceholderText(
            "/path/to/sam_hela/best.pt  (optional; bypasses download)")
        self.ed_model_local.setToolTip(
            "If you already have the SAM-HeLa checkpoint on disk, point\n"
            "the app at it here. Takes precedence over the download URL\n"
            "and stops the app from copying or re-downloading anything.")
        self.btn_model_local_browse = QPushButton("Browse…")
        self.btn_model_local_browse.clicked.connect(self._browse_model_local)

        # ----- Assemble form ----------------------------------------
        layout = QVBoxLayout(self)

        layout.addWidget(out_lbl)
        layout.addWidget(self.rb_sub)
        layout.addWidget(self.rb_pref)
        layout.addWidget(self.rb_cust)
        layout.addLayout(custom_row)
        layout.addWidget(self.lbl_preview)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #444;")
        layout.addWidget(divider)

        layout.addWidget(auto_lbl)
        form = QFormLayout()
        form.addRow("Mode:",          self.combo_autosave)
        form.addRow("Tick interval:", self.spin_interval)
        form.addRow("Min mask flush:", self.spin_mask_min)
        layout.addLayout(form)

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setStyleSheet("color: #444;")
        layout.addWidget(divider2)
        model_lbl = QLabel("SAM-HeLa checkpoint")
        model_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(model_lbl)

        local_row = QHBoxLayout()
        local_row.addWidget(QLabel("Local file:"))
        local_row.addWidget(self.ed_model_local, stretch=1)
        local_row.addWidget(self.btn_model_local_browse)
        layout.addLayout(local_row)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("or URL:    "))
        url_row.addWidget(self.ed_model_url, stretch=1)
        layout.addLayout(url_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_preview()

    # ----- behavior ----------------------------------------------------
    def _selected_mode(self) -> str:
        if self.rb_pref.isChecked():
            return project_io.OUTPUT_MODE_PREFIXED
        if self.rb_cust.isChecked():
            return project_io.OUTPUT_MODE_CUSTOM
        return project_io.OUTPUT_MODE_SUBFOLDER

    def _update_preview(self):
        # Use a placeholder video path so the preview is concrete.
        sample = "/path/to/your/SUM_MC308_..._0009_AAV.avi"
        out = project_io.resolve_output_folder(
            sample, self._selected_mode(), self.ed_custom.text())
        self.lbl_preview.setText(f"Preview: {out}")
        self.ed_custom.setEnabled(self.rb_cust.isChecked())

    def _browse_model_local(self):
        start = self.ed_model_local.text() or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick SAM-HeLa checkpoint", start,
            "PyTorch checkpoint (*.pt *.pth);;All files (*)")
        if path:
            self.ed_model_local.setText(path)

    def _browse_custom_root(self):
        start = os.path.expanduser(self.ed_custom.text() or "~")
        path = QFileDialog.getExistingDirectory(
            self, "Choose custom output root", start)
        if path:
            self.ed_custom.setText(path)
            self.rb_cust.setChecked(True)

    def accept(self):
        from core import model_download
        s = self._settings
        s.setValue(project_io.SETTING_OUTPUT_MODE, self._selected_mode())
        s.setValue(project_io.SETTING_OUTPUT_CUSTOM_ROOT, self.ed_custom.text())
        s.setValue(project_io.SETTING_AUTOSAVE_MODE,
                    self.combo_autosave.currentData())
        s.setValue(project_io.SETTING_AUTOSAVE_INTERVAL_SEC,
                    int(self.spin_interval.value()))
        s.setValue(project_io.SETTING_AUTOSAVE_MASK_MIN_SEC,
                    int(self.spin_mask_min.value()))
        s.setValue(model_download.SETTINGS_KEY, self.ed_model_url.text().strip())
        s.setValue(model_download.LOCAL_PATH_SETTINGS_KEY,
                    self.ed_model_local.text().strip())
        s.sync()
        super().accept()
