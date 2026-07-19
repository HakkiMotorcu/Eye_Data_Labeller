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
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from core import project_io

# ---- Quality-flag settings (Annotation page) ------------------------
QF_ENABLED = 'annotation/quality_flags'
QF_EDGE = 'annotation/flag_edge'
QF_SPLIT = 'annotation/flag_split'
QF_AREA = 'annotation/flag_area'


# ---- Detection settings (Detection page) ----------------------------
DET_CUSTOM = 'detection/custom_thresholds'
DET_PRED_IOU = 'detection/pred_iou'
DET_STABILITY = 'detection/stability'
DET_MIN_PX = 'detection/min_px'
DET_MAX_PX = 'detection/max_px'


def read_detection_settings():
    """{custom, pred_iou, stability, min_px, max_px} for SAM
    auto-segmentation. Thresholds apply only when custom=True; size
    limits of 0 mean 'off'."""
    s = QSettings()

    def _f(key, default):
        try:
            return float(s.value(key, default))
        except (TypeError, ValueError):
            return default

    def _i(key, default):
        try:
            return int(s.value(key, default))
        except (TypeError, ValueError):
            return default

    return {
        'custom': str(s.value(DET_CUSTOM, False)).lower() in ('1', 'true'),
        'pred_iou': _f(DET_PRED_IOU, 0.88),
        'stability': _f(DET_STABILITY, 0.95),
        'min_px': _i(DET_MIN_PX, 0),
        'max_px': _i(DET_MAX_PX, 0),
    }


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

    # Page names in nav order — callers open a specific one by name.
    PAGES = ("Output & Autosave", "SAM Model", "Detection",
             "Annotation", "Debugging")

    def __init__(self, parent=None, page=None):
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
                ("Detection", self._page_detection),
                ("Annotation", self._page_annotation),
                ("Debugging", self._page_debug)):
            self._nav.addItem(name)
            page_w = QWidget()
            lay = QVBoxLayout(page_w)
            builder(lay)
            lay.addStretch(1)
            self._stack.addWidget(page_w)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        # Open on the requested page (by name or index) — e.g. the Model
        # menu opens straight to "SAM Model", not the first tab.
        start = 0
        if isinstance(page, str) and page in self.PAGES:
            start = self.PAGES.index(page)
        elif isinstance(page, int) and 0 <= page < len(self.PAGES):
            start = page
        self._nav.setCurrentRow(start)

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

    # ---- Page: SAM Model (registry editor) -------------------------
    def _page_model(self, lay):
        from core import model_download
        s = self._settings
        model_url = str(s.value(model_download.SETTINGS_KEY,
                                model_download.DEFAULT_SAM_HELA_URL))

        lay.addWidget(_bold("SAM models"))
        lay.addWidget(_hint(
            "Registered checkpoints and built-in variants. Select one and "
            "Make active, or Add / Edit / Remove. Tags and file paths are "
            "each unique. Changes apply immediately."))
        self.list_models = QListWidget()
        self.list_models.setMinimumHeight(170)
        self.list_models.itemDoubleClicked.connect(
            lambda _i: self._model_make_active())
        lay.addWidget(self.list_models)

        btns = QHBoxLayout()
        b_add = QPushButton("Add…"); b_add.clicked.connect(self._model_add)
        b_edit = QPushButton("Edit…"); b_edit.clicked.connect(self._model_edit)
        b_rm = QPushButton("Remove"); b_rm.clicked.connect(self._model_remove)
        b_act = QPushButton("Make active")
        b_act.clicked.connect(self._model_make_active)
        for b in (b_add, b_edit, b_rm):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(b_act)
        lay.addLayout(btns)
        self._refresh_model_list()

        lay.addWidget(_bold("Download URL (built-in variants)"))
        lay.addWidget(_hint(
            "Used to fetch a built-in variant's weights on first use "
            "when no local checkpoint is registered."))
        self.ed_model_url = QLineEdit(model_url)
        self.ed_model_url.setPlaceholderText("https://…/best.pt")
        lay.addWidget(self.ed_model_url)

    def _refresh_model_list(self):
        from core import model_registry
        self.list_models.clear()
        active = model_registry.get_active_tag()
        for m in model_registry.all_models():
            mark = "● " if m['tag'] == active else "   "
            if m.get('builtin'):
                tail = "  (built-in · download)"
            elif m.get('path'):
                tail = f"  ·  {os.path.basename(m['path'])}"
            else:
                tail = "  (no checkpoint)"
            it = QListWidgetItem(f"{mark}{m['tag']}   [{m['base']}]{tail}")
            it.setData(Qt.ItemDataRole.UserRole, m)
            it.setToolTip(m.get('path') or "built-in registry variant")
            self.list_models.addItem(it)

    def _selected_model(self):
        it = self.list_models.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _model_add(self):
        from ui.model_edit_dialog import ModelEditDialog
        from core import model_registry
        res = ModelEditDialog.get_new(self)
        if res is None:
            return
        try:
            entry = model_registry.add_model(res['tag'], res['path'],
                                             res['base'])
        except ValueError as e:
            QMessageBox.warning(self, "Add model", str(e))
            return
        model_registry.set_active(entry['tag'])  # newly added = active
        self._refresh_model_list()

    def _model_edit(self):
        from ui.model_edit_dialog import ModelEditDialog
        from core import model_registry
        m = self._selected_model()
        if m is None:
            return
        if m.get('builtin'):
            QMessageBox.information(
                self, "Edit model",
                "Built-in variants can't be edited — Add a model to "
                "register your own checkpoint.")
            return
        res = ModelEditDialog.get_edit(self, m)
        if res is None:
            return
        try:
            model_registry.update_model(m['tag'], res['tag'], res['path'],
                                        res['base'])
        except ValueError as e:
            QMessageBox.warning(self, "Edit model", str(e))
            return
        self._refresh_model_list()

    def _model_remove(self):
        from core import model_registry
        m = self._selected_model()
        if m is None:
            return
        if m.get('builtin'):
            QMessageBox.information(
                self, "Remove model", "Built-in variants can't be removed.")
            return
        if QMessageBox.question(
                self, "Remove model",
                f"Remove '{m['tag']}' from the registry?\n"
                f"(The checkpoint file on disk is left untouched.)") \
                != QMessageBox.StandardButton.Yes:
            return
        model_registry.remove_model(m['tag'])
        self._refresh_model_list()

    def _model_make_active(self):
        from core import model_registry
        m = self._selected_model()
        if m is None:
            return
        model_registry.set_active(m['tag'])
        self._refresh_model_list()

    # ---- Page: Detection -------------------------------------------
    def _page_detection(self, lay):
        from PyQt6.QtWidgets import QDoubleSpinBox
        det = read_detection_settings()
        lay.addWidget(_bold("SAM auto-segmentation"))
        lay.addWidget(_hint(
            "Tuning for the Auto-segment runs (not SAM Box). Stricter "
            "thresholds find fewer, cleaner cells; looser ones find "
            "more with more noise. Defaults are the model's own."))
        self.chk_det_custom = QCheckBox("Use custom detection thresholds")
        self.chk_det_custom.setChecked(det['custom'])
        lay.addWidget(self.chk_det_custom)

        self.spin_pred_iou = QDoubleSpinBox()
        self.spin_pred_iou.setRange(0.50, 0.95)
        self.spin_pred_iou.setSingleStep(0.01)
        self.spin_pred_iou.setValue(det['pred_iou'])
        self.spin_pred_iou.setToolTip(
            "Predicted mask quality cutoff (model default 0.88).\n"
            "Lower = more detections, more junk.")
        self.spin_stability = QDoubleSpinBox()
        self.spin_stability.setRange(0.50, 0.99)
        self.spin_stability.setSingleStep(0.01)
        self.spin_stability.setValue(det['stability'])
        self.spin_stability.setToolTip(
            "Mask stability cutoff (model default 0.95).\n"
            "Lower = keep masks that shift under threshold changes.")
        form = QFormLayout()
        form.addRow("Quality threshold:", self.spin_pred_iou)
        form.addRow("Stability threshold:", self.spin_stability)
        lay.addLayout(form)
        for w in (self.spin_pred_iou, self.spin_stability):
            self.chk_det_custom.toggled.connect(w.setEnabled)
            w.setEnabled(det['custom'])

        lay.addSpacing(10)
        lay.addWidget(_bold("Size filter"))
        lay.addWidget(_hint(
            "Drop detections outside this pixel-area range before they "
            "become annotations — specks and merged blobs never enter "
            "the list. 0 = no limit. Applies to Auto-segment only."))
        self.spin_det_min = QSpinBox()
        self.spin_det_min.setRange(0, 100000)
        self.spin_det_min.setValue(det['min_px'])
        self.spin_det_min.setSuffix(" px")
        self.spin_det_max = QSpinBox()
        self.spin_det_max.setRange(0, 10000000)
        self.spin_det_max.setValue(det['max_px'])
        self.spin_det_max.setSuffix(" px")
        form2 = QFormLayout()
        form2.addRow("Minimum area:", self.spin_det_min)
        form2.addRow("Maximum area:", self.spin_det_max)
        lay.addLayout(form2)

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
        # Model checkpoints now live in the registry (edited live on the
        # SAM Model page), not a single LOCAL_PATH setting.
        s.setValue(DET_CUSTOM, bool(self.chk_det_custom.isChecked()))
        s.setValue(DET_PRED_IOU, float(self.spin_pred_iou.value()))
        s.setValue(DET_STABILITY, float(self.spin_stability.value()))
        s.setValue(DET_MIN_PX, int(self.spin_det_min.value()))
        s.setValue(DET_MAX_PX, int(self.spin_det_max.value()))
        s.setValue(QF_ENABLED, bool(self.chk_qf.isChecked()))
        s.setValue(QF_EDGE, bool(self.chk_qf_edge.isChecked()))
        s.setValue(QF_SPLIT, bool(self.chk_qf_split.isChecked()))
        s.setValue(QF_AREA, bool(self.chk_qf_area.isChecked()))
        s.setValue(core_debug.SETTING_DEBUG_KEY,
                   bool(self.chk_debug.isChecked()))
        core_debug.set_debug(self.chk_debug.isChecked())  # applies live
        s.sync()
        super().accept()
