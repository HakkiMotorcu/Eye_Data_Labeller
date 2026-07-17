"""Application settings — one panel for every default.

Settings-app layout: category list on the left, the active page on
the right. OK persists everything under the same QSettings keys the
app has always used; Cancel discards.

Pages:
  Output & Autosave   where files go, autosave cadence
  SAM Model           checkpoint path / download URL
  Annotation          quality flags + future annotation defaults
  Debugging           detailed logging, log folder

Adding a future setting = add a row to the right page (or a new page
to _PAGES) and persist it in accept(). Nothing else to wire.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QRadioButton, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from core import project_io

# ---- Quality-flag settings (Annotation page) ------------------------
QF_ENABLED = 'annotation/quality_flags'
QF_EDGE = 'annotation/flag_edge'
QF_SPLIT = 'annotation/flag_split'
QF_AREA = 'annotation/flag_area'


def read_quality_flag_settings():
    """{enabled, edge, split, area} — all default ON (there's an off
    switch in Settings > Annotation, per design)."""
    s = QSettings()

    def _b(key):
        return str(s.value(key, True)).lower() in ('1', 'true')

    return {'enabled': _b(QF_ENABLED), 'edge': _b(QF_EDGE),
            'split': _b(QF_SPLIT), 'area': _b(QF_AREA)}


def _bold(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold;")
    return lbl


def _hint(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #888; font-size: 11px;")
    return lbl


class SettingsDialog(QDialog):
    """Elegant tabless settings panel: nav list + stacked pages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(720, 500)
        self._settings = QSettings()

        self._nav = QListWidget()
        self._nav.setFixedWidth(170)
        self._stack = QStackedWidget()
        for name, builder in (
                ("Output & Autosave", self._page_output),
                ("SAM Model", self._page_model),
                ("Annotation", self._page_annotation),
                ("Debugging", self._page_debug)):
            self._nav.addItem(name)
            page = QWidget()
            lay = QVBoxLayout(page)
            builder(lay)
            lay.addStretch(1)
            self._stack.addWidget(page)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        body = QHBoxLayout()
        body.addWidget(self._nav)
        body.addWidget(self._stack, stretch=1)
        outer = QVBoxLayout(self)
        outer.addLayout(body, stretch=1)
        outer.addWidget(buttons)

    # ---- Page: Output & Autosave -----------------------------------
    def _page_output(self, lay):
        s = self._settings
        mode = str(s.value(project_io.SETTING_OUTPUT_MODE,
                           project_io.DEFAULTS[project_io.SETTING_OUTPUT_MODE]))
        custom_root = str(s.value(
            project_io.SETTING_OUTPUT_CUSTOM_ROOT,
            project_io.DEFAULTS[project_io.SETTING_OUTPUT_CUSTOM_ROOT]))
        autosave_mode = str(s.value(
            project_io.SETTING_AUTOSAVE_MODE,
            project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MODE]))
        try:
            interval_sec = int(s.value(
                project_io.SETTING_AUTOSAVE_INTERVAL_SEC,
                project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_INTERVAL_SEC]))
        except (TypeError, ValueError):
            interval_sec = project_io.DEFAULTS[
                project_io.SETTING_AUTOSAVE_INTERVAL_SEC]
        try:
            mask_min_sec = int(s.value(
                project_io.SETTING_AUTOSAVE_MASK_MIN_SEC,
                project_io.DEFAULTS[project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]))
        except (TypeError, ValueError):
            mask_min_sec = project_io.DEFAULTS[
                project_io.SETTING_AUTOSAVE_MASK_MIN_SEC]

        lay.addWidget(_bold("Output folder"))
        self.rb_sub = QRadioButton("Subfolder under out/ next to the video "
                                   "( <dir>/out/<stem>/ )")
        self.rb_pref = QRadioButton("Folder next to the video "
                                    "( <dir>/<stem>_out/ )")
        self.rb_cust = QRadioButton("Custom root (one folder for every "
                                    "project)")
        grp = QButtonGroup(self)
        for rb in (self.rb_sub, self.rb_pref, self.rb_cust):
            grp.addButton(rb)
            lay.addWidget(rb)
        grp.setExclusive(True)
        {
            project_io.OUTPUT_MODE_SUBFOLDER: self.rb_sub,
            project_io.OUTPUT_MODE_PREFIXED: self.rb_pref,
            project_io.OUTPUT_MODE_CUSTOM: self.rb_cust,
        }.get(mode, self.rb_sub).setChecked(True)

        custom_row = QHBoxLayout()
        self.ed_custom = QLineEdit(custom_root)
        self.ed_custom.setPlaceholderText(
            "…/EyeLabellerProjects   (used when 'Custom root' is selected)")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_custom_root)
        custom_row.addWidget(self.ed_custom, stretch=1)
        custom_row.addWidget(btn_browse)
        lay.addLayout(custom_row)

        self.lbl_preview = _hint("Preview: —")
        self.lbl_preview.setStyleSheet(
            "color: #888; font-family: monospace; font-size: 11px;")
        lay.addWidget(self.lbl_preview)
        for rb in (self.rb_sub, self.rb_pref, self.rb_cust):
            rb.toggled.connect(self._update_preview)
        self.ed_custom.textChanged.connect(self._update_preview)

        lay.addSpacing(10)
        lay.addWidget(_bold("Auto-save"))
        self.combo_autosave = QComboBox()
        self.combo_autosave.addItem("Off", project_io.AUTOSAVE_OFF)
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
        form = QFormLayout()
        form.addRow("Mode:", self.combo_autosave)
        form.addRow("Tick interval:", self.spin_interval)
        form.addRow("Min mask flush:", self.spin_mask_min)
        lay.addLayout(form)
        self._update_preview()

    # ---- Page: SAM Model -------------------------------------------
    def _page_model(self, lay):
        from core import model_download
        s = self._settings
        model_url = str(s.value(model_download.SETTINGS_KEY,
                                model_download.DEFAULT_SAM_HELA_URL))
        model_local = str(s.value(model_download.LOCAL_PATH_SETTINGS_KEY, ""))

        lay.addWidget(_bold("SAM-HeLa checkpoint"))
        lay.addWidget(_hint(
            "The fine-tuned weights (best.pt) that power SAM Box. A local "
            "file takes precedence; the URL is used to download on first "
            "use when no file is set."))
        self.ed_model_local = QLineEdit(model_local)
        self.ed_model_local.setPlaceholderText(
            "/path/to/sam_hela/best.pt  (optional; bypasses download)")
        btn_local = QPushButton("Browse…")
        btn_local.clicked.connect(self._browse_model_local)
        row = QHBoxLayout()
        row.addWidget(QLabel("Local file:"))
        row.addWidget(self.ed_model_local, stretch=1)
        row.addWidget(btn_local)
        lay.addLayout(row)

        self.ed_model_url = QLineEdit(model_url)
        self.ed_model_url.setPlaceholderText(
            "https://huggingface.co/…/best.pt   "
            "(downloaded on first use if not already on disk)")
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("or URL:    "))
        row2.addWidget(self.ed_model_url, stretch=1)
        lay.addLayout(row2)

    # ---- Page: Annotation ------------------------------------------
    def _page_annotation(self, lay):
        qf = read_quality_flag_settings()
        lay.addWidget(_bold("Quality flags"))
        lay.addWidget(_hint(
            "Suspicious annotations get a ⚠ in the list (hover the row "
            "for the reasons). Pure geometry checks — deterministic and "
            "explainable; the flag only says 'look at me', you decide."))
        self.chk_qf = QCheckBox("Enable quality flags")
        self.chk_qf.setChecked(qf['enabled'])
        lay.addWidget(self.chk_qf)
        self.chk_qf_edge = QCheckBox(
            "Mask touches its bbox edge (probable SAM spill-over)")
        self.chk_qf_edge.setChecked(qf['edge'])
        self.chk_qf_split = QCheckBox(
            "Mask split into disconnected pieces (one 'cell', two blobs)")
        self.chk_qf_split.setChecked(qf['split'])
        self.chk_qf_area = QCheckBox(
            "Area is an outlier vs the frame's median cell (>4× or <¼×)")
        self.chk_qf_area.setChecked(qf['area'])
        for chk in (self.chk_qf_edge, self.chk_qf_split, self.chk_qf_area):
            chk.setStyleSheet("margin-left: 18px;")
            lay.addWidget(chk)
            self.chk_qf.toggled.connect(chk.setEnabled)
            chk.setEnabled(qf['enabled'])

    # ---- Page: Debugging -------------------------------------------
    def _page_debug(self, lay):
        from core import debug as core_debug
        lay.addWidget(_bold("Debugging"))
        self.chk_debug = QCheckBox(
            "Detailed logging — record every action to the log file")
        self.chk_debug.setChecked(core_debug.is_debug())
        self.chk_debug.setToolTip(
            "Errors are always logged. Turn this on to also record every\n"
            "action (frame changes, SAM runs, saves, brush strokes …) —\n"
            "then send the newest file from the log folder when reporting\n"
            "a problem. Applies immediately; persists across launches.")
        lay.addWidget(self.chk_debug)
        btn_logs = QPushButton("Open log folder…")
        btn_logs.clicked.connect(self._open_log_folder)
        cur = core_debug.log_file_path()
        row = QHBoxLayout()
        row.addWidget(btn_logs)
        row.addWidget(_hint(f"This session: {os.path.basename(cur)}" if cur
                            else "This session: no log file yet"), stretch=1)
        lay.addLayout(row)

    # ---- Behavior ----------------------------------------------------
    def _selected_mode(self) -> str:
        if self.rb_pref.isChecked():
            return project_io.OUTPUT_MODE_PREFIXED
        if self.rb_cust.isChecked():
            return project_io.OUTPUT_MODE_CUSTOM
        return project_io.OUTPUT_MODE_SUBFOLDER

    def _update_preview(self):
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

    def _open_log_folder(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        from core import debug as core_debug
        QDesktopServices.openUrl(QUrl.fromLocalFile(core_debug.log_dir()))

    def accept(self):
        from core import model_download
        from core import debug as core_debug
        s = self._settings
        s.setValue(project_io.SETTING_OUTPUT_MODE, self._selected_mode())
        s.setValue(project_io.SETTING_OUTPUT_CUSTOM_ROOT,
                   self.ed_custom.text())
        s.setValue(project_io.SETTING_AUTOSAVE_MODE,
                   self.combo_autosave.currentData())
        s.setValue(project_io.SETTING_AUTOSAVE_INTERVAL_SEC,
                   int(self.spin_interval.value()))
        s.setValue(project_io.SETTING_AUTOSAVE_MASK_MIN_SEC,
                   int(self.spin_mask_min.value()))
        s.setValue(model_download.SETTINGS_KEY,
                   self.ed_model_url.text().strip())
        s.setValue(model_download.LOCAL_PATH_SETTINGS_KEY,
                   self.ed_model_local.text().strip())
        s.setValue(QF_ENABLED, bool(self.chk_qf.isChecked()))
        s.setValue(QF_EDGE, bool(self.chk_qf_edge.isChecked()))
        s.setValue(QF_SPLIT, bool(self.chk_qf_split.isChecked()))
        s.setValue(QF_AREA, bool(self.chk_qf_area.isChecked()))
        s.setValue(core_debug.SETTING_DEBUG_KEY,
                   bool(self.chk_debug.isChecked()))
        core_debug.set_debug(self.chk_debug.isChecked())  # applies live
        s.sync()
        super().accept()
