from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QGroupBox, QListWidget,
                             QFileDialog, QMessageBox, QSlider, QScrollArea,
                             QSizePolicy, QSpinBox, QButtonGroup, QSplitter,
                             QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import os


class FilterSection(QWidget):
    """Collapsible section used in the View panel.

    Header has: chevron, title, active-state dot, and a small reset button.
    Body is hidden when the section is collapsed.
    """

    reset_requested = pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = False
        self._active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.setSpacing(0)

        self._header = QWidget()
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            "QWidget{background:transparent;}"
            "QWidget:hover{background:#2a2a30;}")
        hlay = QHBoxLayout(self._header)
        hlay.setContentsMargins(4, 3, 4, 3)
        hlay.setSpacing(6)

        self._lbl_arrow = QLabel("▾")
        self._lbl_arrow.setStyleSheet("color:#888;font-size:11px;")
        self._lbl_arrow.setFixedWidth(12)
        hlay.addWidget(self._lbl_arrow)

        self._lbl_title = QLabel(title)
        self._lbl_title.setStyleSheet("color:#cfd0d3;font-size:11px;")
        hlay.addWidget(self._lbl_title)

        hlay.addStretch(1)

        self._lbl_dot = QLabel("●")
        self._lbl_dot.setStyleSheet("color:#3a3a40;font-size:11px;")
        self._lbl_dot.setToolTip("Inactive")
        hlay.addWidget(self._lbl_dot)

        self.btn_reset = QPushButton()
        self.btn_reset.setFixedSize(20, 20)
        self.btn_reset.setToolTip(f"Reset {title} to defaults")
        self.btn_reset.setFlat(True)
        try:
            import qtawesome as qta
            self.btn_reset.setIcon(qta.icon('fa6s.rotate-left', color='#888'))
        except Exception:
            self.btn_reset.setText("↺")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        hlay.addWidget(self.btn_reset)

        outer.addWidget(self._header)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 2, 4, 4)
        self._body_layout.setSpacing(3)
        outer.addWidget(self._body)

        # Toggle collapse on header click (but not on the reset button).
        self._header.mousePressEvent = self._on_header_click

    def _on_header_click(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if not self.btn_reset.geometry().contains(ev.pos()):
                self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed):
        self._collapsed = bool(collapsed)
        self._body.setVisible(not self._collapsed)
        self._lbl_arrow.setText("▸" if self._collapsed else "▾")

    def set_active(self, active):
        self._active = bool(active)
        color = "#4cc26a" if self._active else "#3a3a40"
        self._lbl_dot.setStyleSheet(f"color:{color};font-size:11px;")
        self._lbl_dot.setToolTip("Active" if self._active else "Inactive")

    def add_layout(self, lay):
        self._body_layout.addLayout(lay)

    def add_widget(self, w):
        self._body_layout.addWidget(w)


class MainWindow(QMainWindow):
    def __init__(self, video_data=None):
        super().__init__()
        self.resize(1500, 900)
        self.video_data = video_data
        self.seg_data = None
        self._current_file = None
        self._current_frame_idx = 0

        # ----- View state (display-only enhancement pipeline) -----
        # These never mutate the underlying frame data. They only affect
        # what is composed into the pyqtgraph ImageItem.
        self._projection_mode = 'none'      # 'none' | 'std' | 'max' | 'mean' | 'sum' | 'min'
        self._projection_cache = None       # (H, W) uint8
        self._projection_cache_key = None   # tuple — invalidates cache when changed
        self._proj_window_mode = 'all'      # 'all' | 'sliding' | 'range'
        self._proj_sliding_n = 3
        self._proj_range_lo = 0
        self._proj_range_hi = 0
        self._proj_percentile_clip = False
        self._bg_subtract_on = False
        self._bg_subtract_window = 2
        self._clahe_on = False
        self._clahe_clip = 2.0
        self._clahe_tile = 8
        self._frangi_on = False
        self._frangi_sigma_min = 1.0
        self._frangi_sigma_max = 4.0
        self._frangi_n_sigmas = 4
        self._frangi_black_ridges = False
        self._gamma = 1.0
        self._invert = False

        pg.setConfigOptions(imageAxisOrder='row-major')
        self._setup_ui()

        # Wire every View-panel control to the same re-render path.
        self.btn_reset_view.clicked.connect(self._reset_view_filters)
        # Per-section reset buttons (the ↺ icon in each section header)
        self.section_projection.reset_requested.connect(self._reset_projection_section)
        self.section_bg.reset_requested.connect(self._reset_bg_section)
        self.section_clahe.reset_requested.connect(self._reset_clahe_section)
        self.section_frangi.reset_requested.connect(self._reset_frangi_section)
        self.section_lut.reset_requested.connect(self._reset_lut_section)
        self.combo_projection.currentTextChanged.connect(self._on_projection_changed)
        self.combo_proj_window.currentTextChanged.connect(self._on_proj_window_mode_changed)
        self.slider_proj_sliding.valueChanged.connect(self._on_proj_sliding_changed)
        self.spin_proj_range_lo.valueChanged.connect(self._on_proj_range_lo_changed)
        self.spin_proj_range_hi.valueChanged.connect(self._on_proj_range_hi_changed)
        self.chk_proj_clip.toggled.connect(self._on_proj_clip_toggled)
        self.chk_bg_subtract.toggled.connect(self._on_bg_subtract_toggled)
        self.slider_bg_window.valueChanged.connect(self._on_bg_window_changed)
        self.chk_clahe.toggled.connect(self._on_clahe_toggled)
        self.slider_clahe_clip.valueChanged.connect(self._on_clahe_clip_changed)
        self.slider_clahe_tile.valueChanged.connect(self._on_clahe_tile_changed)
        self.chk_frangi.toggled.connect(self._on_frangi_toggled)
        self.chk_frangi_black.toggled.connect(self._on_frangi_black_toggled)
        self.slider_frangi_smin.valueChanged.connect(self._on_frangi_smin_changed)
        self.slider_frangi_smax.valueChanged.connect(self._on_frangi_smax_changed)
        self.slider_frangi_n.valueChanged.connect(self._on_frangi_n_changed)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        self.chk_invert.toggled.connect(self._on_invert_toggled)

        if self.video_data:
            self.load_video()
        self._update_title()

    @staticmethod
    def _filter_divider(title):
        """Section divider used between filter families in the View panel."""
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(4)
        line_left = QFrame()
        line_left.setFrameShape(QFrame.Shape.HLine)
        line_left.setStyleSheet("color:#2c2c33;")
        line_left.setFixedWidth(8)
        lbl = QLabel(title)
        lbl.setStyleSheet("color:#888;font-size:10px;")
        line_right = QFrame()
        line_right.setFrameShape(QFrame.Shape.HLine)
        line_right.setStyleSheet("color:#2c2c33;")
        lay.addWidget(line_left)
        lay.addWidget(lbl)
        lay.addWidget(line_right, stretch=1)
        return w

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Left: Frame view ---
        frame_container = QWidget()
        frame_container.setMinimumWidth(400)
        frame_layout = QVBoxLayout(frame_container)
        frame_layout.setContentsMargins(5, 5, 5, 5)

        self.lbl_frame_title = QLabel("Frame View")
        self.lbl_frame_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_title.setStyleSheet("font-weight: bold; color: #555;")
        frame_layout.addWidget(self.lbl_frame_title)

        self.view_frame = self._create_image_view()
        frame_layout.addWidget(self.view_frame)

        # Segmentation overlay — transparent RGBA ImageItem on top of the frame
        self._seg_overlay = pg.ImageItem()
        self._seg_overlay.setZValue(10)  # above the frame image
        self.view_frame.getView().addItem(self._seg_overlay)
        self._seg_visible = True

        # --- Right: Scrollable Panels (draggable splitter on the left edge) ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(600)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right_panel_widget = QWidget()
        right_panel = QVBoxLayout(right_panel_widget)
        right_panel.setContentsMargins(4, 4, 4, 4)
        right_panel.setSpacing(4)

        _COMPACT_SS = ("QGroupBox{padding-top:14px;margin-top:4px}"
                       "QGroupBox::title{subcontrol-origin:margin;left:6px}")

        # 1. Annotations (formerly Inspector + Annotations, merged)
        list_group = QGroupBox("Annotations")
        list_group.setStyleSheet(_COMPACT_SS)
        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(4, 2, 4, 4)
        list_layout.setSpacing(2)

        self.lbl_coords = QLabel("No annotation selected")
        self.lbl_coords.setStyleSheet("font-family: monospace; color: #888; font-size: 11px;")
        self.lbl_coords.setWordWrap(True)
        list_layout.addWidget(self.lbl_coords)

        self.lbl_stats = QLabel("0 annotations")
        self.lbl_stats.setStyleSheet("font-family: monospace; color: #aaa; font-size: 10px;")
        self.lbl_stats.setWordWrap(True)
        list_layout.addWidget(self.lbl_stats)

        self.list_annotations = QListWidget()
        self.list_annotations.setMinimumHeight(60)
        self.list_annotations.setMaximumHeight(150)
        self.list_annotations.setEditTriggers(
            self.list_annotations.EditTrigger.DoubleClicked)
        list_layout.addWidget(self.list_annotations)

        # Row 1: create — Cell / Vessel / Capillary
        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self.btn_add = QPushButton("Cell [A]")
        self.btn_add.setToolTip("Add a cell annotation (bbox + segmentation) (A)")
        self.btn_add_vessel = QPushButton("Vessel [V]")
        self.btn_add_vessel.setToolTip("Add a vessel — paint-only, no bbox (V)")
        self.btn_add_vessel.setStyleSheet("color: #9370db;")
        self.btn_add_capillary = QPushButton("Capillary [C]")
        self.btn_add_capillary.setToolTip("Add a capillary — paint-only, no bbox (C)")
        self.btn_add_capillary.setStyleSheet("color: #eb82c8;")
        add_row.addWidget(self.btn_add)
        add_row.addWidget(self.btn_add_vessel)
        add_row.addWidget(self.btn_add_capillary)
        list_layout.addLayout(add_row)

        # Row 2: modify — Delete / Rename / Fit BBox
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        self.btn_delete = QPushButton("Del [Del]")
        self.btn_delete.setToolTip("Delete the selected annotation")
        self.btn_delete.setStyleSheet("color: #e76f51;")
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setToolTip("Rename selected annotation (double-click list item)")
        self.btn_fit_bbox = QPushButton("Fit BBox")
        self.btn_fit_bbox.setToolTip("Fit bbox to actual seg pixels")
        actions_row.addWidget(self.btn_delete)
        actions_row.addWidget(self.btn_rename)
        actions_row.addWidget(self.btn_fit_bbox)
        list_layout.addLayout(actions_row)

        list_group.setLayout(list_layout)

        # 3. Tools
        tools_group = self._make_collapsible_group("Tools", _COMPACT_SS)
        tools_layout = QVBoxLayout()
        tools_layout.setContentsMargins(4, 2, 4, 4)
        tools_layout.setSpacing(3)

        lock_layout = QHBoxLayout()
        lock_layout.setSpacing(4)
        self.btn_lock = QPushButton("Lock [L]")
        self.btn_unlock = QPushButton("Unlock [U]")
        lock_layout.addWidget(self.btn_lock)
        lock_layout.addWidget(self.btn_unlock)

        lock_all_layout = QHBoxLayout()
        lock_all_layout.setSpacing(4)
        self.btn_lock_all = QPushButton("Lock All")
        self.btn_unlock_all = QPushButton("Unlock All")
        lock_all_layout.addWidget(self.btn_lock_all)
        lock_all_layout.addWidget(self.btn_unlock_all)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(4)
        self.btn_hide_locked = QPushButton("Hide Locked BBoxes [H]")
        self.btn_hide_locked.setCheckable(True)
        self.btn_hide_locked.setToolTip(
            "Hide bboxes of locked annotations so they don't get in the way\n"
            "when labeling neighboring cells (H)")
        toggle_row.addWidget(self.btn_hide_locked)

        toggle_row2 = QHBoxLayout()
        toggle_row2.setSpacing(4)
        self.btn_label_colors = QPushButton("Status Colors")
        self.btn_label_colors.setCheckable(True)
        self.btn_label_colors.setToolTip(
            "Simple coloring: Red=unlocked, Yellow=selected, Green=locked")
        toggle_row2.addWidget(self.btn_label_colors)

        tools_layout.addLayout(lock_layout)
        tools_layout.addLayout(lock_all_layout)
        tools_layout.addLayout(toggle_row)
        tools_layout.addLayout(toggle_row2)

        # --- Seg Editing controls (merged into Tools panel) ---
        from PyQt6.QtWidgets import QFrame
        _seg_divider = QFrame()
        _seg_divider.setFrameShape(QFrame.Shape.HLine)
        _seg_divider.setStyleSheet("color:#444")
        tools_layout.addWidget(_seg_divider)
        _seg_label = QLabel("Seg editing")
        _seg_label.setStyleSheet("color:#888; font-size:10px;")
        tools_layout.addWidget(_seg_label)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(3)
        self.btn_mode_select = QPushButton("Select")
        self.btn_mode_select.setCheckable(True)
        self.btn_mode_select.setChecked(True)
        self.btn_mode_select.setToolTip("Normal selection mode (Esc)")
        self.btn_mode_paint = QPushButton("Paint [D]")
        self.btn_mode_paint.setCheckable(True)
        self.btn_mode_paint.setToolTip("Paint seg mask pixels (D)")
        self.btn_mode_erase = QPushButton("Erase [E]")
        self.btn_mode_erase.setCheckable(True)
        self.btn_mode_erase.setToolTip("Erase seg mask pixels (E)")
        self._seg_mode_group = QButtonGroup(self)
        self._seg_mode_group.setExclusive(True)
        self._seg_mode_group.addButton(self.btn_mode_select, 0)
        self._seg_mode_group.addButton(self.btn_mode_paint, 1)
        self._seg_mode_group.addButton(self.btn_mode_erase, 2)
        mode_row.addWidget(self.btn_mode_select)
        mode_row.addWidget(self.btn_mode_paint)
        mode_row.addWidget(self.btn_mode_erase)
        tools_layout.addLayout(mode_row)

        # Brush size
        brush_row = QHBoxLayout()
        brush_row.setSpacing(4)
        brush_row.addWidget(QLabel("Brush"))
        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(1, 30)
        self.slider_brush_size.setValue(5)
        brush_row.addWidget(self.slider_brush_size, stretch=1)
        self.lbl_brush_size = QLabel("5")
        self.lbl_brush_size.setFixedWidth(20)
        self.lbl_brush_size.setStyleSheet("font-family: monospace; color: #aaa;")
        brush_row.addWidget(self.lbl_brush_size)
        tools_layout.addLayout(brush_row)

        # Action buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(3)
        self.btn_fill_bbox = QPushButton("Fill BBox [F]")
        self.btn_fill_bbox.setToolTip("Fill selected annotation's bbox as seg instance (F)")
        self.btn_save_seg = QPushButton("Save Seg")
        self.btn_save_seg.setToolTip("Export modified segmentation masks as AVI")
        self.btn_save_seg.setStyleSheet("color: #2a9d8f; font-weight: bold;")
        action_row.addWidget(self.btn_fill_bbox)
        action_row.addWidget(self.btn_save_seg)
        tools_layout.addLayout(action_row)

        # Force paint toggle (bypasses safe-paint mode)
        force_row = QHBoxLayout()
        force_row.setSpacing(3)
        self.btn_force_paint = QPushButton("Force Paint [X]")
        self.btn_force_paint.setCheckable(True)
        self.btn_force_paint.setChecked(False)
        self.btn_force_paint.setToolTip(
            "UNCHECKED (safe): paint only on empty space — never overwrites other instances.\n"
            "CHECKED (force): paint overwrites every pixel, including other instances.\n"
            "Use force mode only to deliberately overwrite / correct misassigned pixels.")
        self.btn_force_paint.setStyleSheet(
            "QPushButton:checked { color: #e76f51; font-weight: bold; }"
            "QPushButton:!checked { color: #2a9d8f; }")
        force_row.addWidget(self.btn_force_paint)
        tools_layout.addLayout(force_row)

        # Propagate mask row
        propagate_row = QHBoxLayout()
        propagate_row.setSpacing(3)
        self.btn_propagate_mask = QPushButton("Propagate Mask  [Ctrl+P]")
        self.btn_propagate_mask.setToolTip(
            "Copy this frame's painted pixels to all other frames that share\n"
            "this annotation (same instance).\n\n"
            "Frames already painted are skipped by default — you will be asked\n"
            "whether to overwrite them.\n\n"
            "Workflow for veins:\n"
            "  1. Add Vein → Yes to propagate across all frames\n"
            "  2. Paint the vein mask on one frame\n"
            "  3. Press Ctrl+P to copy it to every other frame\n"
            "  4. Navigate frame-by-frame to adjust where needed")
        self.btn_propagate_mask.setStyleSheet("color: #9370db; font-weight: bold;")
        propagate_row.addWidget(self.btn_propagate_mask)
        tools_layout.addLayout(propagate_row)

        tools_group.setLayout(tools_layout)

        # 3. SAM (Segment Anything assisted labeling) -----------------
        sam_group = self._make_collapsible_group("SAM", _COMPACT_SS)
        sam_layout = QVBoxLayout()
        sam_layout.setContentsMargins(4, 2, 4, 4)
        sam_layout.setSpacing(3)

        # Model selector
        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        model_row.addWidget(QLabel("Model"))
        self.combo_sam_model = QComboBox()
        self.combo_sam_model.addItems([
            "sam_hela (fine-tuned ViT-B)",
            "vit_b_lm (light microscopy)",
            "vit_t (mobile, fastest)",
            "vit_b (SAM base)",
            "vit_l (SAM large)",
        ])
        self.combo_sam_model.setToolTip(
            "Which SAM variant to load.\n"
            "sam_hela is the collaborators' fine-tuned checkpoint at\n"
            "models/checkpoints/sam_hela/best.pt — the default.")
        model_row.addWidget(self.combo_sam_model, stretch=1)
        sam_layout.addLayout(model_row)

        # Status line — driven by SamService state
        self.lbl_sam_status = QLabel("model: not loaded")
        self.lbl_sam_status.setStyleSheet(
            "font-family: monospace; color: #888; font-size: 10px;")
        self.lbl_sam_status.setWordWrap(True)
        sam_layout.addWidget(self.lbl_sam_status)

        # Embedding precompute — wires up in Phase 4.2
        emb_row = QHBoxLayout()
        emb_row.setSpacing(4)
        self.btn_sam_precompute = QPushButton("Precompute embeddings")
        self.btn_sam_precompute.setToolTip(
            "Pre-encode every frame so subsequent prompts are interactive.\n"
            "First time may take 1–2 s/frame on CPU, ~0.1 s on GPU.\n"
            "Result cached to {video}_sam_embeddings.zarr next to the image.\n"
            "(Wires up in Phase 4.2 — currently a placeholder.)")
        self.btn_sam_precompute.setEnabled(False)  # enabled when 4.2 lands
        emb_row.addWidget(self.btn_sam_precompute)
        sam_layout.addLayout(emb_row)

        # Scope selector + duplicate guard
        scope_row = QHBoxLayout()
        scope_row.setSpacing(4)
        self.chk_sam_all_frames = QCheckBox("All frames")
        self.chk_sam_all_frames.setToolTip(
            "Off (default): Auto-segment runs on the current frame only.\n"
            "On: loop over every frame in the stack — the status line shows\n"
            "progress; UI may freeze briefly between frames (no async yet).")
        self.chk_sam_avoid_dupes = QCheckBox("Avoid duplicates")
        self.chk_sam_avoid_dupes.setChecked(True)
        self.chk_sam_avoid_dupes.setToolTip(
            "On (default): if a SAM detection overlaps an existing annotation\n"
            "by IoU > 0.30 on the same frame, ABSORB it (paint SAM's pixels\n"
            "into the existing instance_id, keep the existing bbox shape) or\n"
            "DROP it (when the existing annotation already has painted pixels).\n"
            "Off: every SAM detection becomes a new annotation — may duplicate\n"
            "manually-labeled cells.")
        scope_row.addWidget(self.chk_sam_all_frames)
        scope_row.addWidget(self.chk_sam_avoid_dupes)
        scope_row.addStretch(1)
        sam_layout.addLayout(scope_row)

        # Run SAM — automatic instance segmentation.
        # ADDITIVE: existing annotations + seg pixels are preserved; new
        # cells get fresh instance_ids and gap-filled names.
        run_row = QHBoxLayout()
        run_row.setSpacing(4)
        self.btn_run_sam = QPushButton("Auto-segment")
        self.btn_run_sam.setToolTip(
            "Run automatic instance segmentation on the raw frame data.\n"
            "Additive — does NOT delete existing annotations.\n"
            "SAM-found cells get fresh IDs and Cell_N names (gap-filled).")
        self.btn_run_sam.setStyleSheet("color: #4cc9f0; font-weight: bold;")
        run_row.addWidget(self.btn_run_sam)
        sam_layout.addLayout(run_row)

        sam_group.setLayout(sam_layout)

        # 4. File I/O
        io_group = self._make_collapsible_group("Import / Export", _COMPACT_SS)
        io_layout = QVBoxLayout()
        io_layout.setContentsMargins(4, 2, 4, 4)
        io_layout.setSpacing(4)

        # Format picker + Load Seg on one row
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(4)
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItems(["CSV", "JSON"])
        self.combo_export_format.setToolTip("Output format for all export buttons")
        self.btn_load_seg = QPushButton("Load Seg")
        self.btn_load_seg.setToolTip("Load instance segmentation AVI")
        self.btn_load_seg.setStyleSheet("color: #e9c46a; font-weight: bold;")
        fmt_row.addWidget(QLabel("Format:"))
        fmt_row.addWidget(self.combo_export_format, stretch=1)
        fmt_row.addWidget(self.btn_load_seg)
        io_layout.addLayout(fmt_row)

        # Export options checkboxes
        opts_row = QHBoxLayout()
        opts_row.setSpacing(6)
        self.chk_export_bbox = QCheckBox("BBoxes")
        self.chk_export_bbox.setChecked(True)
        self.chk_export_bbox.setToolTip(
            "Include bounding-box columns (x0, y0, width, height) in export")
        self.chk_export_seg_bbox = QCheckBox("Seg BBoxes")
        self.chk_export_seg_bbox.setChecked(False)
        self.chk_export_seg_bbox.setToolTip(
            "Re-compute tight bounding boxes from the painted seg mask pixels\n"
            "and use those instead of the ROI box.\n"
            "Useful after brush edits changed the actual shape.")
        self.chk_export_vein_flag = QCheckBox("Vein flag")
        self.chk_export_vein_flag.setChecked(True)
        self.chk_export_vein_flag.setToolTip(
            "Add an 'inside_vein' column (1/0) to the cells export:\n"
            "1 = cell centroid or bbox overlaps a vein mask pixel on that frame.")
        opts_row.addWidget(self.chk_export_bbox)
        opts_row.addWidget(self.chk_export_seg_bbox)
        opts_row.addWidget(self.chk_export_vein_flag)
        io_layout.addLayout(opts_row)

        # Export buttons: Cells / Veins / All
        export_row = QHBoxLayout()
        export_row.setSpacing(4)
        self.btn_export_cells = QPushButton("Export Cells")
        self.btn_export_cells.setToolTip(
            "Export only cell annotations (class_type='cell') to a file.")
        self.btn_export_cells.setStyleSheet("color: #2a9d8f; font-weight: bold;")
        self.btn_export_veins = QPushButton("Export Veins")
        self.btn_export_veins.setToolTip(
            "Export only vein annotations (class_type='vein') to a file.")
        self.btn_export_veins.setStyleSheet("color: #9370db; font-weight: bold;")
        self.btn_export_all = QPushButton("Export All")
        self.btn_export_all.setToolTip(
            "Export all annotations (cells + veins) to a single file.")
        self.btn_export_all.setStyleSheet("color: #e9c46a; font-weight: bold;")
        export_row.addWidget(self.btn_export_cells)
        export_row.addWidget(self.btn_export_veins)
        export_row.addWidget(self.btn_export_all)
        io_layout.addLayout(export_row)

        # COCO sidecar export — the modern, training-pipeline-friendly format.
        coco_row = QHBoxLayout()
        coco_row.setSpacing(4)
        self.btn_export_coco = QPushButton("Export COCO")
        self.btn_export_coco.setToolTip(
            "Write a COCO JSON sidecar (RLE masks + bboxes + categories).\n"
            "Ready to drop into Detectron2 / MMDetection / Roboflow / "
            "pycocotools.")
        self.btn_export_coco.setStyleSheet("color: #4cc9f0; font-weight: bold;")
        coco_row.addWidget(self.btn_export_coco)
        io_layout.addLayout(coco_row)

        # Import row
        import_row = QHBoxLayout()
        import_row.setSpacing(4)
        self.btn_import = QPushButton("Import Annotations  [Ctrl+I]")
        self.btn_import.setStyleSheet("color: #457b9d; font-weight: bold;")
        import_row.addWidget(self.btn_import)
        io_layout.addLayout(import_row)

        io_group.setLayout(io_layout)

        # 4. View (display + segmentation overlay; Phase 3 will add image-enhancement controls here)
        display_group = self._make_collapsible_group("View", _COMPACT_SS)
        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(4, 2, 4, 4)
        display_layout.setSpacing(2)

        cmap_layout = QHBoxLayout()
        cmap_layout.setSpacing(4)
        cmap_layout.addWidget(QLabel("Colormap"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems([
            "gray", "magma", "viridis", "inferno", "plasma",
            "hot", "cool", "CET-L1",
        ])
        cmap_layout.addWidget(self.combo_colormap, stretch=1)
        display_layout.addLayout(cmap_layout)

        # ==============================================================
        # FILTER PIPELINE (display-only, applied in this order):
        #   raw or projection or bg-subtract  ->  CLAHE  ->  Frangi
        #   ->  gamma  ->  invert  ->  display
        # Every control re-renders the current frame through this chain.
        # ==============================================================

        # Top strip: Reset View + live pipeline summary
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        self.lbl_pipeline = QLabel("Pipeline: identity")
        self.lbl_pipeline.setStyleSheet(
            "font-family: monospace; color: #cfd0d3; font-size: 10px;"
            "background: #1f1f24; padding: 3px 6px; border-radius: 3px;")
        self.lbl_pipeline.setWordWrap(True)
        self.lbl_pipeline.setToolTip(
            "Effective display pipeline (in order): every active filter "
            "and its key parameter.")
        header_row.addWidget(self.lbl_pipeline, stretch=1)
        self.btn_reset_view = QPushButton("Reset")
        self.btn_reset_view.setFixedWidth(60)
        self.btn_reset_view.setToolTip(
            "Return every filter to its default state.\n"
            "Individual sections also have their own reset (↺) button.")
        header_row.addWidget(self.btn_reset_view)
        display_layout.addLayout(header_row)

        # ---- Projection (collapsible) ---------------------------------
        proj_row = QHBoxLayout()
        proj_row.setSpacing(4)
        proj_row.addWidget(QLabel("Mode"))
        self.combo_projection = QComboBox()
        self.combo_projection.addItems(["None", "Std", "Max", "Mean", "Sum", "Min"])
        self.combo_projection.setToolTip(
            "Reduction across the selected frame window.\n"
            "Std is the headline mode for AOSLO — stationary vessel walls\n"
            "drop out, moving cells pop.")
        proj_row.addWidget(self.combo_projection, stretch=1)

        win_mode_row = QHBoxLayout()
        win_mode_row.setSpacing(4)
        win_mode_row.addWidget(QLabel("Window"))
        self.combo_proj_window = QComboBox()
        self.combo_proj_window.addItems(["All frames", "Sliding ±N", "Range"])
        self.combo_proj_window.setToolTip(
            "All frames: project across the whole stack (cached, one image).\n"
            "Sliding ±N: project around the current frame — projection "
            "follows the timeline.\n"
            "Range: project over a fixed [lo..hi] frame range.")
        win_mode_row.addWidget(self.combo_proj_window, stretch=1)

        # Sliding ±N slider (visible only in 'sliding' mode)
        self.proj_sliding_row = QWidget()
        slide_lay = QHBoxLayout(self.proj_sliding_row)
        slide_lay.setContentsMargins(0, 0, 0, 0)
        slide_lay.setSpacing(4)
        slide_lay.addWidget(QLabel("± N"))
        self.slider_proj_sliding = QSlider(Qt.Orientation.Horizontal)
        self.slider_proj_sliding.setRange(1, 30)
        self.slider_proj_sliding.setValue(3)
        self.slider_proj_sliding.setToolTip(
            "Half-width of the sliding window around the current frame.")
        slide_lay.addWidget(self.slider_proj_sliding, stretch=1)
        self.lbl_proj_sliding = QLabel("3")
        self.lbl_proj_sliding.setFixedWidth(28)
        self.lbl_proj_sliding.setStyleSheet("font-family: monospace; color: #aaa;")
        slide_lay.addWidget(self.lbl_proj_sliding)
        self.proj_sliding_row.setVisible(False)

        # Range [lo..hi] spinboxes (visible only in 'range' mode)
        self.proj_range_row = QWidget()
        rng_lay = QHBoxLayout(self.proj_range_row)
        rng_lay.setContentsMargins(0, 0, 0, 0)
        rng_lay.setSpacing(4)
        rng_lay.addWidget(QLabel("Range"))
        self.spin_proj_range_lo = QSpinBox()
        self.spin_proj_range_lo.setRange(0, 0)
        self.spin_proj_range_lo.setPrefix("lo: ")
        rng_lay.addWidget(self.spin_proj_range_lo)
        self.spin_proj_range_hi = QSpinBox()
        self.spin_proj_range_hi.setRange(0, 0)
        self.spin_proj_range_hi.setPrefix("hi: ")
        rng_lay.addWidget(self.spin_proj_range_hi)
        rng_lay.addStretch(1)
        self.proj_range_row.setVisible(False)

        # Percentile-clip toggle for normalization
        clip_row = QHBoxLayout()
        clip_row.setSpacing(4)
        self.chk_proj_clip = QCheckBox("Percentile clip (1–99%)")
        self.chk_proj_clip.setToolTip(
            "Normalize the projection using the 1st..99th percentile of its\n"
            "values instead of full min/max. Stops a single hot pixel from\n"
            "washing out the displayed contrast.")
        clip_row.addWidget(self.chk_proj_clip)
        clip_row.addStretch(1)

        # Build the Projection section
        self.section_projection = FilterSection("Projection")
        self.section_projection.add_layout(proj_row)
        self.section_projection.add_layout(win_mode_row)
        self.section_projection.add_widget(self.proj_sliding_row)
        self.section_projection.add_widget(self.proj_range_row)
        self.section_projection.add_layout(clip_row)
        display_layout.addWidget(self.section_projection)

        # ---- Background subtraction ----------------------------------
        bg_row = QHBoxLayout()
        bg_row.setSpacing(4)
        self.chk_bg_subtract = QCheckBox("Enabled")
        self.chk_bg_subtract.setToolTip(
            "Subtract a rolling-mean background around the current frame.\n"
            "Surfaces slow-moving immune cells inside a stationary vessel.")
        bg_row.addWidget(self.chk_bg_subtract)
        bg_row.addWidget(QLabel("± win"))
        self.slider_bg_window = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_window.setRange(1, 10)
        self.slider_bg_window.setValue(2)
        self.slider_bg_window.setToolTip("Half-width of the rolling-mean window")
        bg_row.addWidget(self.slider_bg_window, stretch=1)
        self.lbl_bg_window = QLabel("2")
        self.lbl_bg_window.setFixedWidth(28)
        self.lbl_bg_window.setStyleSheet("font-family: monospace; color: #aaa;")
        bg_row.addWidget(self.lbl_bg_window)
        self.section_bg = FilterSection("Background subtraction")
        self.section_bg.add_layout(bg_row)
        display_layout.addWidget(self.section_bg)

        # ---- CLAHE ---------------------------------------------------
        clahe_row = QHBoxLayout()
        clahe_row.setSpacing(4)
        self.chk_clahe = QCheckBox("Enabled")
        self.chk_clahe.setToolTip("Contrast-Limited Adaptive Histogram Equalization")
        clahe_row.addWidget(self.chk_clahe)
        clahe_row.addStretch(1)

        clahe_clip_row = QHBoxLayout()
        clahe_clip_row.setSpacing(4)
        clahe_clip_row.addWidget(QLabel("Clip"))
        self.slider_clahe_clip = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe_clip.setRange(5, 100)      # mapped to 0.5 .. 10.0
        self.slider_clahe_clip.setValue(20)          # default 2.0
        self.slider_clahe_clip.setToolTip(
            "Clip limit. Higher = stronger local contrast.")
        clahe_clip_row.addWidget(self.slider_clahe_clip, stretch=1)
        self.lbl_clahe_clip = QLabel("2.0")
        self.lbl_clahe_clip.setFixedWidth(36)
        self.lbl_clahe_clip.setStyleSheet("font-family: monospace; color: #aaa;")
        clahe_clip_row.addWidget(self.lbl_clahe_clip)

        clahe_tile_row = QHBoxLayout()
        clahe_tile_row.setSpacing(4)
        clahe_tile_row.addWidget(QLabel("Tiles"))
        self.slider_clahe_tile = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe_tile.setRange(2, 16)
        self.slider_clahe_tile.setValue(8)
        self.slider_clahe_tile.setToolTip(
            "Number of tiles across the image; smaller = more local.")
        clahe_tile_row.addWidget(self.slider_clahe_tile, stretch=1)
        self.lbl_clahe_tile = QLabel("8")
        self.lbl_clahe_tile.setFixedWidth(28)
        self.lbl_clahe_tile.setStyleSheet("font-family: monospace; color: #aaa;")
        clahe_tile_row.addWidget(self.lbl_clahe_tile)

        self.section_clahe = FilterSection("CLAHE (contrast)")
        self.section_clahe.add_layout(clahe_row)
        self.section_clahe.add_layout(clahe_clip_row)
        self.section_clahe.add_layout(clahe_tile_row)
        display_layout.addWidget(self.section_clahe)

        # ---- Frangi vesselness ---------------------------------------
        frangi_row = QHBoxLayout()
        frangi_row.setSpacing(4)
        self.chk_frangi = QCheckBox("Enabled")
        self.chk_frangi.setToolTip(
            "Multi-scale Frangi vesselness response.\n"
            "Slow filter — ~1 sec for a 500×500 frame; tweak sigmas to taste.")
        self.chk_frangi_black = QCheckBox("Dark ridges")
        self.chk_frangi_black.setToolTip(
            "Off (default): bright vessels on dark background (AOSLO).\n"
            "On: dark vessels on bright background (fundus-style).")
        frangi_row.addWidget(self.chk_frangi)
        frangi_row.addWidget(self.chk_frangi_black)
        frangi_row.addStretch(1)

        sigmin_row = QHBoxLayout()
        sigmin_row.setSpacing(4)
        sigmin_row.addWidget(QLabel("σ min"))
        self.slider_frangi_smin = QSlider(Qt.Orientation.Horizontal)
        self.slider_frangi_smin.setRange(5, 100)     # 0.5 .. 10.0
        self.slider_frangi_smin.setValue(10)         # 1.0
        self.slider_frangi_smin.setToolTip("Smallest vessel radius (px)")
        sigmin_row.addWidget(self.slider_frangi_smin, stretch=1)
        self.lbl_frangi_smin = QLabel("1.0")
        self.lbl_frangi_smin.setFixedWidth(36)
        self.lbl_frangi_smin.setStyleSheet("font-family: monospace; color: #aaa;")
        sigmin_row.addWidget(self.lbl_frangi_smin)

        sigmax_row = QHBoxLayout()
        sigmax_row.setSpacing(4)
        sigmax_row.addWidget(QLabel("σ max"))
        self.slider_frangi_smax = QSlider(Qt.Orientation.Horizontal)
        self.slider_frangi_smax.setRange(10, 200)    # 1.0 .. 20.0
        self.slider_frangi_smax.setValue(40)         # 4.0
        self.slider_frangi_smax.setToolTip("Largest vessel radius (px)")
        sigmax_row.addWidget(self.slider_frangi_smax, stretch=1)
        self.lbl_frangi_smax = QLabel("4.0")
        self.lbl_frangi_smax.setFixedWidth(36)
        self.lbl_frangi_smax.setStyleSheet("font-family: monospace; color: #aaa;")
        sigmax_row.addWidget(self.lbl_frangi_smax)

        nsig_row = QHBoxLayout()
        nsig_row.setSpacing(4)
        nsig_row.addWidget(QLabel("Scales"))
        self.slider_frangi_n = QSlider(Qt.Orientation.Horizontal)
        self.slider_frangi_n.setRange(1, 8)
        self.slider_frangi_n.setValue(4)
        self.slider_frangi_n.setToolTip(
            "Number of scales sampled between σ min and σ max.\n"
            "More scales = slower but smoother across vessel widths.")
        nsig_row.addWidget(self.slider_frangi_n, stretch=1)
        self.lbl_frangi_n = QLabel("4")
        self.lbl_frangi_n.setFixedWidth(28)
        self.lbl_frangi_n.setStyleSheet("font-family: monospace; color: #aaa;")
        nsig_row.addWidget(self.lbl_frangi_n)

        self.section_frangi = FilterSection("Vesselness (Frangi)")
        self.section_frangi.add_layout(frangi_row)
        self.section_frangi.add_layout(sigmin_row)
        self.section_frangi.add_layout(sigmax_row)
        self.section_frangi.add_layout(nsig_row)
        display_layout.addWidget(self.section_frangi)

        # ---- Display LUT (gamma + invert) ----------------------------
        lut_row = QHBoxLayout()
        lut_row.setSpacing(4)
        lut_row.addWidget(QLabel("Gamma"))
        self.slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setRange(10, 400)
        self.slider_gamma.setValue(100)
        self.slider_gamma.setToolTip("Display gamma (1.00 = identity)")
        lut_row.addWidget(self.slider_gamma, stretch=1)
        self.lbl_gamma = QLabel("1.00")
        self.lbl_gamma.setFixedWidth(40)
        self.lbl_gamma.setStyleSheet("font-family: monospace; color: #aaa;")
        lut_row.addWidget(self.lbl_gamma)
        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.setToolTip("Invert display values (255 - x)")
        lut_row.addWidget(self.chk_invert)

        self.section_lut = FilterSection("Display LUT")
        self.section_lut.add_layout(lut_row)
        display_layout.addWidget(self.section_lut)

        level_grid = QHBoxLayout()
        level_grid.setSpacing(4)
        min_col = QVBoxLayout()
        min_col.setSpacing(0)
        min_col.addWidget(QLabel("Min"))
        self.slider_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_min.setRange(0, 255)
        self.slider_min.setValue(0)
        min_col.addWidget(self.slider_min)
        max_col = QVBoxLayout()
        max_col.setSpacing(0)
        max_col.addWidget(QLabel("Max"))
        self.slider_max = QSlider(Qt.Orientation.Horizontal)
        self.slider_max.setRange(0, 255)
        self.slider_max.setValue(255)
        max_col.addWidget(self.slider_max)
        level_grid.addLayout(min_col)
        level_grid.addLayout(max_col)
        display_layout.addLayout(level_grid)

        level_info_row = QHBoxLayout()
        self.lbl_levels = QLabel("Levels: 0 - 255")
        self.lbl_levels.setStyleSheet("font-family: monospace; color: #888; font-size: 10px;")
        self.btn_auto_levels = QPushButton("Auto")
        self.btn_auto_levels.setToolTip("Reset levels to data min/max")
        self.btn_auto_levels.setFixedWidth(50)
        level_info_row.addWidget(self.lbl_levels, stretch=1)
        level_info_row.addWidget(self.btn_auto_levels)
        display_layout.addLayout(level_info_row)

        # Seg overlay opacity
        seg_opacity_layout = QHBoxLayout()
        seg_opacity_layout.setSpacing(4)
        seg_opacity_layout.addWidget(QLabel("Seg opacity"))
        self.slider_seg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_seg_opacity.setRange(0, 100)
        self.slider_seg_opacity.setValue(40)
        seg_opacity_layout.addWidget(self.slider_seg_opacity, stretch=1)
        self.btn_toggle_seg = QPushButton("Seg [S]")
        self.btn_toggle_seg.setCheckable(True)
        self.btn_toggle_seg.setChecked(True)
        self.btn_toggle_seg.setToolTip("Toggle segmentation overlay visibility (S)")
        seg_opacity_layout.addWidget(self.btn_toggle_seg)
        display_layout.addLayout(seg_opacity_layout)

        display_group.setLayout(display_layout)

        # Assemble right panel
        right_panel.addWidget(list_group)       # Annotations
        right_panel.addWidget(tools_group)      # Tools (incl. seg editing)
        right_panel.addWidget(sam_group)        # SAM (model + auto + future prompts)
        right_panel.addWidget(display_group)    # View
        right_panel.addWidget(io_group)         # I/O (Phase 2 will rework)
        right_panel.addStretch(1)

        scroll_area.setWidget(right_panel_widget)

        # Horizontal splitter so the user can drag-resize the divider
        # between the frame view and the right panel. Sensible minimums
        # on each side prevent the user from collapsing either to zero.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(frame_container)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 1)   # frame view absorbs window growth
        splitter.setStretchFactor(1, 0)   # sidebar stays at its size
        splitter.setSizes([1100, 380])    # initial split
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle{background:#2c2c33;}"
            "QSplitter::handle:hover{background:#3d6ea8;}"
        )

        main_layout.addWidget(splitter, stretch=1)

        # --- Bottom: Timeline ---
        timeline_group = QGroupBox()
        timeline_group.setStyleSheet("QGroupBox{padding:2px;margin:0}")
        timeline_layout = QHBoxLayout(timeline_group)
        timeline_layout.setContentsMargins(8, 2, 8, 2)
        timeline_layout.setSpacing(6)

        self.btn_frame_first = QPushButton("\u23EE")
        self.btn_frame_first.setFixedWidth(30)
        self.btn_frame_prev = QPushButton("\u25C0")
        self.btn_frame_prev.setFixedWidth(30)

        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setValue(0)

        self.btn_frame_next = QPushButton("\u25B6")
        self.btn_frame_next.setFixedWidth(30)
        self.btn_frame_last = QPushButton("\u23ED")
        self.btn_frame_last.setFixedWidth(30)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setPrefix("Frame: ")
        self.spin_frame.setFixedWidth(120)

        self.lbl_total_frames = QLabel("/ 0")
        self.lbl_total_frames.setStyleSheet("color: #888; font-family: monospace;")

        timeline_layout.addWidget(self.btn_frame_first)
        timeline_layout.addWidget(self.btn_frame_prev)
        timeline_layout.addWidget(self.slider_timeline, stretch=1)
        timeline_layout.addWidget(self.btn_frame_next)
        timeline_layout.addWidget(self.btn_frame_last)
        timeline_layout.addWidget(self.spin_frame)
        timeline_layout.addWidget(self.lbl_total_frames)

        main_layout.addWidget(timeline_group)

        # --- Bottom status bar (napari-style: frame / coords / pixel value) ---
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.setStyleSheet(
            "QStatusBar{background:#1f1f24;border-top:1px solid #2c2c33;}"
            "QStatusBar::item{border:0;}"
        )

        self.lbl_status_image = QLabel("no image")
        self.lbl_status_frame = QLabel("Frame: — / —")
        self.lbl_status_coords = QLabel("(—, —)")
        self.lbl_status_value = QLabel("val: —")
        for w in (self.lbl_status_image, self.lbl_status_frame,
                  self.lbl_status_coords, self.lbl_status_value):
            w.setStyleSheet("font-family: monospace; color: #bbb; padding: 0 10px;")
        # Add a leading info label on the left, others on the right
        sb.addWidget(self.lbl_status_image)
        sb.addPermanentWidget(self.lbl_status_frame)
        sb.addPermanentWidget(self.lbl_status_coords)
        sb.addPermanentWidget(self.lbl_status_value)

        # Pyqtgraph mouse-move signal — drives coords/value readout.
        self.view_frame.scene.sigMouseMoved.connect(self._on_mouse_moved)

        # Decorate buttons with FontAwesome icons (no-op without qtawesome).
        self._apply_icons()

    # ------------------------------------------------------------------
    # DISPLAY PIPELINE — composes the image fed to pyqtgraph from the raw
    # frame plus any active enhancements. The underlying FrameSource is
    # never mutated; enhancements affect display only.
    # ------------------------------------------------------------------
    def _compose_display_frame(self, idx):
        # Step 1: base image — projection trumps live frame; bg-subtract
        # is per-frame so it can sit on top of either.
        if self._bg_subtract_on:
            base = self._bg_subtract_for_frame(idx)
        elif self._projection_mode != 'none':
            cached = self._get_projection()
            base = cached if cached is not None else self.video_data.get_frame(idx)
        else:
            base = self.video_data.get_frame(idx)

        # Step 2: enhancement filters with their tuned parameters.
        if self._clahe_on:
            from core import enhance
            base = enhance.apply_clahe(
                base,
                clip_limit=self._clahe_clip,
                tile_grid_size=self._clahe_tile,
            )
        if self._frangi_on:
            from core import enhance
            base = enhance.frangi_vesselness(
                base,
                sigma_min=self._frangi_sigma_min,
                sigma_max=self._frangi_sigma_max,
                n_sigmas=self._frangi_n_sigmas,
                black_ridges=self._frangi_black_ridges,
            )

        # Step 3: display LUT (gamma + invert).
        if abs(self._gamma - 1.0) > 1e-3:
            from core import enhance
            base = enhance.apply_gamma(base, self._gamma)
        if self._invert:
            base = 255 - base

        return base

    def _bg_subtract_for_frame(self, idx):
        import numpy as np
        from core import motion
        frames = getattr(self.video_data, 'frames', None)
        if frames is None:
            frames = np.stack([self.video_data.get_frame(i)
                               for i in range(self.video_data.num_frames)])
        return motion.bg_subtract(frames, idx, self._bg_subtract_window)

    def _projection_cache_key_now(self):
        """Tuple identifying the inputs the cached projection depends on.

        For Sliding-window mode, the cache must invalidate every time the
        current frame moves; for All / Range it persists across frames.
        """
        ctr = self._current_frame_idx if self._proj_window_mode == 'sliding' else None
        return (self._projection_mode, self._proj_window_mode, self._proj_sliding_n,
                self._proj_range_lo, self._proj_range_hi,
                self._proj_percentile_clip, ctr)

    def _get_projection(self):
        if not self.video_data:
            return None
        key = self._projection_cache_key_now()
        if self._projection_cache is not None and key == self._projection_cache_key:
            return self._projection_cache
        from core import projections
        frames = getattr(self.video_data, 'frames', None)
        if frames is None:
            import numpy as np
            frames = np.stack([self.video_data.get_frame(i)
                               for i in range(self.video_data.num_frames)])
        self._projection_cache = projections.project_stack(
            frames, self._projection_mode,
            window_mode=self._proj_window_mode,
            window=self._proj_sliding_n,
            center=self._current_frame_idx,
            range_lo=self._proj_range_lo,
            range_hi=self._proj_range_hi,
            percentile_clip=self._proj_percentile_clip,
        )
        self._projection_cache_key = key
        return self._projection_cache

    def _rerender(self):
        self._refresh_view_status()
        if self.video_data:
            self.display_frame(self._current_frame_idx, auto_range=False)

    def _refresh_view_status(self):
        """Update each section's active-state dot and the pipeline strip."""
        proj_active = self._projection_mode != 'none'
        lut_active = abs(self._gamma - 1.0) > 1e-3 or self._invert
        self.section_projection.set_active(proj_active)
        self.section_bg.set_active(self._bg_subtract_on)
        self.section_clahe.set_active(self._clahe_on)
        self.section_frangi.set_active(self._frangi_on)
        self.section_lut.set_active(lut_active)

        parts = []
        if proj_active:
            tag = self._projection_mode.upper()
            if self._proj_window_mode == 'sliding':
                tag += f"±{self._proj_sliding_n}"
            elif self._proj_window_mode == 'range':
                tag += f"[{self._proj_range_lo}..{self._proj_range_hi}]"
            if self._proj_percentile_clip:
                tag += " clip"
            parts.append(tag)
        if self._bg_subtract_on:
            parts.append(f"BG±{self._bg_subtract_window}")
        if self._clahe_on:
            parts.append(f"CLAHE({self._clahe_clip:.1f})")
        if self._frangi_on:
            parts.append(
                f"FRANGI[{self._frangi_sigma_min:.1f}..{self._frangi_sigma_max:.1f}]"
                + (" dark" if self._frangi_black_ridges else ""))
        if abs(self._gamma - 1.0) > 1e-3:
            parts.append(f"γ={self._gamma:.2f}")
        if self._invert:
            parts.append("INV")
        self.lbl_pipeline.setText(
            "Pipeline: " + (" · ".join(parts) if parts else "identity"))

    def _on_projection_changed(self, text):
        self._projection_mode = (text or 'none').lower()
        self._projection_cache = None
        self._rerender()

    def _on_proj_window_mode_changed(self, text):
        mode_map = {"All frames": 'all', "Sliding ±N": 'sliding', "Range": 'range'}
        self._proj_window_mode = mode_map.get(text, 'all')
        self.proj_sliding_row.setVisible(self._proj_window_mode == 'sliding')
        self.proj_range_row.setVisible(self._proj_window_mode == 'range')
        self._projection_cache = None
        if self._projection_mode != 'none':
            self._rerender()

    def _on_proj_sliding_changed(self, value):
        self._proj_sliding_n = int(value)
        self.lbl_proj_sliding.setText(str(int(value)))
        if self._projection_mode != 'none' and self._proj_window_mode == 'sliding':
            self._projection_cache = None
            self._rerender()

    def _on_proj_range_lo_changed(self, value):
        self._proj_range_lo = int(value)
        if self._proj_range_lo > self._proj_range_hi:
            self.spin_proj_range_hi.blockSignals(True)
            self.spin_proj_range_hi.setValue(self._proj_range_lo)
            self.spin_proj_range_hi.blockSignals(False)
            self._proj_range_hi = self._proj_range_lo
        if self._projection_mode != 'none' and self._proj_window_mode == 'range':
            self._projection_cache = None
            self._rerender()

    def _on_proj_range_hi_changed(self, value):
        self._proj_range_hi = int(value)
        if self._proj_range_hi < self._proj_range_lo:
            self.spin_proj_range_lo.blockSignals(True)
            self.spin_proj_range_lo.setValue(self._proj_range_hi)
            self.spin_proj_range_lo.blockSignals(False)
            self._proj_range_lo = self._proj_range_hi
        if self._projection_mode != 'none' and self._proj_window_mode == 'range':
            self._projection_cache = None
            self._rerender()

    def _on_proj_clip_toggled(self, checked):
        self._proj_percentile_clip = bool(checked)
        self._projection_cache = None
        if self._projection_mode != 'none':
            self._rerender()

    def _on_bg_subtract_toggled(self, checked):
        self._bg_subtract_on = bool(checked)
        self._rerender()

    def _on_bg_window_changed(self, value):
        self._bg_subtract_window = int(value)
        self.lbl_bg_window.setText(str(int(value)))
        if self._bg_subtract_on:
            self._rerender()

    def _on_clahe_toggled(self, checked):
        self._clahe_on = bool(checked)
        self._rerender()

    def _on_clahe_clip_changed(self, value):
        self._clahe_clip = value / 10.0
        self.lbl_clahe_clip.setText(f"{self._clahe_clip:.1f}")
        if self._clahe_on:
            self._rerender()

    def _on_clahe_tile_changed(self, value):
        self._clahe_tile = int(value)
        self.lbl_clahe_tile.setText(str(int(value)))
        if self._clahe_on:
            self._rerender()

    def _on_frangi_toggled(self, checked):
        self._frangi_on = bool(checked)
        self._rerender()

    def _on_frangi_black_toggled(self, checked):
        self._frangi_black_ridges = bool(checked)
        if self._frangi_on:
            self._rerender()

    def _on_frangi_smin_changed(self, value):
        self._frangi_sigma_min = value / 10.0
        self.lbl_frangi_smin.setText(f"{self._frangi_sigma_min:.1f}")
        # Keep min ≤ max to avoid weird Frangi output.
        if self._frangi_sigma_min > self._frangi_sigma_max:
            self.slider_frangi_smax.blockSignals(True)
            self.slider_frangi_smax.setValue(int(self._frangi_sigma_min * 10))
            self.slider_frangi_smax.blockSignals(False)
            self._frangi_sigma_max = self._frangi_sigma_min
            self.lbl_frangi_smax.setText(f"{self._frangi_sigma_max:.1f}")
        if self._frangi_on:
            self._rerender()

    def _on_frangi_smax_changed(self, value):
        self._frangi_sigma_max = value / 10.0
        self.lbl_frangi_smax.setText(f"{self._frangi_sigma_max:.1f}")
        if self._frangi_sigma_max < self._frangi_sigma_min:
            self.slider_frangi_smin.blockSignals(True)
            self.slider_frangi_smin.setValue(int(self._frangi_sigma_max * 10))
            self.slider_frangi_smin.blockSignals(False)
            self._frangi_sigma_min = self._frangi_sigma_max
            self.lbl_frangi_smin.setText(f"{self._frangi_sigma_min:.1f}")
        if self._frangi_on:
            self._rerender()

    def _on_frangi_n_changed(self, value):
        self._frangi_n_sigmas = int(value)
        self.lbl_frangi_n.setText(str(int(value)))
        if self._frangi_on:
            self._rerender()

    def _on_gamma_changed(self, value):
        self._gamma = max(0.1, value / 100.0)
        self.lbl_gamma.setText(f"{self._gamma:.2f}")
        self._rerender()

    def _on_invert_toggled(self, checked):
        self._invert = bool(checked)
        self._rerender()

    # ----- Per-section resets --------------------------------------------
    def _reset_projection_section(self):
        for w, v in [(self.combo_projection, 0),
                     (self.combo_proj_window, 0),
                     (self.slider_proj_sliding, 3),
                     (self.spin_proj_range_lo, 0),
                     (self.spin_proj_range_hi, max(0, (self.video_data.num_frames - 1) if self.video_data else 0)),
                     (self.chk_proj_clip, False)]:
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(v)
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._projection_mode = 'none'
        self._proj_window_mode = 'all'
        self._proj_sliding_n = 3
        self._proj_range_lo = 0
        self._proj_range_hi = max(0, (self.video_data.num_frames - 1) if self.video_data else 0)
        self._proj_percentile_clip = False
        self._projection_cache = None
        self.proj_sliding_row.setVisible(False)
        self.proj_range_row.setVisible(False)
        self.lbl_proj_sliding.setText("3")
        self._rerender()

    def _reset_bg_section(self):
        for w, v in [(self.chk_bg_subtract, False), (self.slider_bg_window, 2)]:
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(False)
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._bg_subtract_on = False
        self._bg_subtract_window = 2
        self.lbl_bg_window.setText("2")
        self._rerender()

    def _reset_clahe_section(self):
        for w, v in [(self.chk_clahe, False),
                     (self.slider_clahe_clip, 20),
                     (self.slider_clahe_tile, 8)]:
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(False)
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._clahe_on = False
        self._clahe_clip = 2.0
        self._clahe_tile = 8
        self.lbl_clahe_clip.setText("2.0")
        self.lbl_clahe_tile.setText("8")
        self._rerender()

    def _reset_frangi_section(self):
        for w, v in [(self.chk_frangi, False),
                     (self.chk_frangi_black, False),
                     (self.slider_frangi_smin, 10),
                     (self.slider_frangi_smax, 40),
                     (self.slider_frangi_n, 4)]:
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._frangi_on = False
        self._frangi_black_ridges = False
        self._frangi_sigma_min = 1.0
        self._frangi_sigma_max = 4.0
        self._frangi_n_sigmas = 4
        self.lbl_frangi_smin.setText("1.0")
        self.lbl_frangi_smax.setText("4.0")
        self.lbl_frangi_n.setText("4")
        self._rerender()

    def _reset_lut_section(self):
        for w, v in [(self.slider_gamma, 100), (self.chk_invert, False)]:
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(False)
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._gamma = 1.0
        self._invert = False
        self.lbl_gamma.setText("1.00")
        self._rerender()

    def _reset_view_filters(self):
        """Return every View-panel filter to its default state.

        Block signals during the slider/checkbox updates so we don't fire
        a re-render per widget — single re-render at the end.
        """
        n_frames = self.video_data.num_frames if self.video_data else 1
        defaults = [
            (self.combo_projection, 0),
            (self.combo_proj_window, 0),
            (self.slider_proj_sliding, 3),
            (self.spin_proj_range_lo, 0),
            (self.spin_proj_range_hi, max(0, n_frames - 1)),
            (self.chk_proj_clip, False),
            (self.chk_bg_subtract, False),
            (self.slider_bg_window, 2),
            (self.chk_clahe, False),
            (self.slider_clahe_clip, 20),
            (self.slider_clahe_tile, 8),
            (self.chk_frangi, False),
            (self.chk_frangi_black, False),
            (self.slider_frangi_smin, 10),
            (self.slider_frangi_smax, 40),
            (self.slider_frangi_n, 4),
            (self.slider_gamma, 100),
            (self.chk_invert, False),
        ]
        for w, val in defaults:
            w.blockSignals(True)
            if hasattr(w, 'setChecked'):
                w.setChecked(bool(val))
            elif hasattr(w, 'setCurrentIndex'):
                w.setCurrentIndex(val)
            else:
                w.setValue(val)
            w.blockSignals(False)

        self._projection_mode = 'none'
        self._projection_cache = None
        self._projection_cache_key = None
        self._proj_window_mode = 'all'
        self._proj_sliding_n = 3
        self._proj_range_lo = 0
        self._proj_range_hi = max(0, n_frames - 1)
        self._proj_percentile_clip = False
        self.proj_sliding_row.setVisible(False)
        self.proj_range_row.setVisible(False)
        self._bg_subtract_on = False
        self._bg_subtract_window = 2
        self._clahe_on = False
        self._clahe_clip = 2.0
        self._clahe_tile = 8
        self._frangi_on = False
        self._frangi_sigma_min = 1.0
        self._frangi_sigma_max = 4.0
        self._frangi_n_sigmas = 4
        self._frangi_black_ridges = False
        self._gamma = 1.0
        self._invert = False

        # Sync the live numeric labels.
        self.lbl_proj_sliding.setText("3")
        self.lbl_bg_window.setText("2")
        self.lbl_clahe_clip.setText("2.0")
        self.lbl_clahe_tile.setText("8")
        self.lbl_frangi_smin.setText("1.0")
        self.lbl_frangi_smax.setText("4.0")
        self.lbl_frangi_n.setText("4")
        self.lbl_gamma.setText("1.00")

        self._rerender()

    def _apply_icons(self):
        """Attach FontAwesome icons to common-action buttons.

        No-op if qtawesome is not installed — every button still shows its
        text label, so the UI remains usable.
        """
        try:
            import qtawesome as qta
        except ImportError:
            return
        color = '#cfd0d3'
        icon_map = {
            'btn_add':           'fa6s.plus',
            'btn_add_vessel':    'fa6s.plus',
            'btn_add_capillary': 'fa6s.plus',
            'btn_delete':        'fa6s.trash',
            'btn_rename':        'fa6s.pen-to-square',
            'btn_fit_bbox':      'fa6s.compress',
            'btn_lock':          'fa6s.lock',
            'btn_unlock':        'fa6s.lock-open',
            'btn_lock_all':      'fa6s.lock',
            'btn_unlock_all':    'fa6s.lock-open',
            'btn_hide_locked':   'fa6s.eye-slash',
            'btn_label_colors':  'fa6s.palette',
            'btn_mode_select':   'fa6s.arrow-pointer',
            'btn_mode_paint':    'fa6s.paintbrush',
            'btn_mode_erase':    'fa6s.eraser',
            'btn_fill_bbox':     'fa6s.fill-drip',
            'btn_force_paint':   'fa6s.bolt',
            'btn_propagate_mask':'fa6s.arrows-left-right',
            'btn_save_seg':      'fa6s.floppy-disk',
            'btn_load_seg':      'fa6s.folder-open',
            'btn_run_sam':       'fa6s.wand-magic-sparkles',
            'btn_import':        'fa6s.file-import',
            'btn_export_cells':  'fa6s.file-export',
            'btn_export_veins':  'fa6s.file-export',
            'btn_export_all':    'fa6s.file-export',
            'btn_auto_levels':   'fa6s.gauge-high',
            'btn_toggle_seg':    'fa6s.eye',
            'btn_frame_first':   'fa6s.backward-fast',
            'btn_frame_prev':    'fa6s.backward-step',
            'btn_frame_next':    'fa6s.forward-step',
            'btn_frame_last':    'fa6s.forward-fast',
        }
        for attr, icon_name in icon_map.items():
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            try:
                btn.setIcon(qta.icon(icon_name, color=color))
            except Exception:
                pass

    def _on_mouse_moved(self, scene_pos):
        if not self.video_data:
            return
        view = self.view_frame.getView()
        if view is None:
            return
        pt = view.mapSceneToView(scene_pos)
        x, y = int(pt.x()), int(pt.y())
        if 0 <= x < self.video_data.width and 0 <= y < self.video_data.height:
            frame = self.video_data.get_frame(self._current_frame_idx)
            val = int(frame[y, x])
            self.lbl_status_coords.setText(f"(x={x}, y={y})")
            self.lbl_status_value.setText(f"val: {val}")
        else:
            self.lbl_status_coords.setText("(—, —)")
            self.lbl_status_value.setText("val: —")

    @staticmethod
    def _make_collapsible_group(title, base_ss=""):
        grp = QGroupBox(title)
        grp.setCheckable(True)
        grp.setChecked(True)
        grp.setStyleSheet(base_ss)

        def _toggle(checked):
            for i in range(grp.layout().count()) if grp.layout() else []:
                item = grp.layout().itemAt(i)
                w = item.widget()
                if w:
                    w.setVisible(checked)
                elif item.layout():
                    _set_layout_visible(item.layout(), checked)

        def _set_layout_visible(lay, visible):
            for i in range(lay.count()):
                item = lay.itemAt(i)
                w = item.widget()
                if w:
                    w.setVisible(visible)
                elif item.layout():
                    _set_layout_visible(item.layout(), visible)

        grp.toggled.connect(_toggle)
        return grp

    def _create_image_view(self, show_histogram=False):
        view = pg.ImageView()
        view.ui.roiBtn.hide()
        view.ui.menuBtn.hide()
        if not show_histogram:
            view.ui.histogram.hide()
        return view

    def get_colormap(self):
        name = self.combo_colormap.currentText()
        cmap = None
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            pass
        if cmap is None:
            try:
                cmap = pg.colormap.get(name, source='matplotlib')
            except Exception:
                pass
        if cmap is None:
            # Build a simple grayscale colormap as fallback
            cmap = pg.ColorMap([0.0, 1.0],
                               [(0, 0, 0, 255), (255, 255, 255, 255)])
        return cmap

    # ------------------------------------------------------------------
    # VIDEO LOADING / FRAME DISPLAY
    # ------------------------------------------------------------------
    def load_video(self):
        """Initialize display after video data is loaded."""
        if not self.video_data:
            return
        self._current_frame_idx = 0

        n = self.video_data.num_frames
        self.slider_timeline.setRange(0, n - 1)
        self.slider_timeline.setValue(0)
        self.spin_frame.setRange(0, n - 1)
        self.spin_frame.setValue(0)
        self.lbl_total_frames.setText(f"/ {n - 1}")

        # Projection range defaults to the whole stack.
        for spin in (self.spin_proj_range_lo, self.spin_proj_range_hi):
            spin.blockSignals(True)
            spin.setRange(0, n - 1)
            spin.blockSignals(False)
        self.spin_proj_range_lo.setValue(0)
        self.spin_proj_range_hi.setValue(n - 1)
        self._proj_range_lo = 0
        self._proj_range_hi = n - 1
        self._projection_cache = None
        # Cap sliding ±N at half the stack so it can never exceed it.
        self.slider_proj_sliding.setMaximum(max(1, n // 2 + 1))

        self._init_level_sliders()
        self.display_frame(0, auto_range=True)

    def display_frame(self, idx, auto_range=False):
        """Display a specific frame in the view."""
        if not self.video_data:
            return
        idx = max(0, min(idx, self.video_data.num_frames - 1))
        self._current_frame_idx = idx

        frame = self._compose_display_frame(idx)
        cmap = self.get_colormap()
        self.view_frame.setImage(frame, autoLevels=False, autoRange=auto_range)
        self.view_frame.setColorMap(cmap)

        lo = self.slider_min.value()
        hi = self.slider_max.value()
        self._apply_levels(lo, hi)

        # Update segmentation overlay
        self._update_seg_overlay()

        self.lbl_frame_title.setText(
            f"Frame {idx} / {self.video_data.num_frames - 1}  "
            f"({self.video_data.width} x {self.video_data.height})")
        # Status bar mirror
        if hasattr(self, 'lbl_status_frame'):
            self.lbl_status_frame.setText(
                f"Frame: {idx} / {self.video_data.num_frames - 1}")
            if self._current_file:
                self.lbl_status_image.setText(
                    f"{os.path.basename(self._current_file)} — "
                    f"{self.video_data.width}×{self.video_data.height}")

    def _update_title(self):
        base = "Eye Data Labeller"
        if self._current_file:
            base += f"  \u2014  {os.path.basename(self._current_file)}"
        self.setWindowTitle(base)

    def get_export_default_name(self, ext=".csv", fmt_tag=""):
        if self._current_file:
            stem = os.path.splitext(os.path.basename(self._current_file))[0]
            return os.path.join(os.path.dirname(self._current_file),
                                f"{stem}_annotations{fmt_tag}{ext}")
        return os.path.join(os.getcwd(), f"annotations{fmt_tag}{ext}")

    # ------------------------------------------------------------------
    # DISPLAY / LEVEL CONTROLS
    # ------------------------------------------------------------------
    def _init_level_sliders(self):
        if not self.video_data:
            return
        frame = self.video_data.get_frame(0)
        dmin = int(np.min(frame))
        dmax = int(np.max(frame))
        slider_max = max(dmax, 1)

        self.slider_min.blockSignals(True)
        self.slider_max.blockSignals(True)
        self.slider_min.setRange(dmin, slider_max)
        self.slider_max.setRange(dmin, slider_max)
        self.slider_min.setValue(dmin)
        self.slider_max.setValue(slider_max)
        self.slider_min.blockSignals(False)
        self.slider_max.blockSignals(False)

        self._apply_levels(dmin, slider_max)

    def _apply_levels(self, lo, hi):
        if hi <= lo:
            hi = lo + 1
        self.lbl_levels.setText(f"Levels: {lo} - {hi}")
        self.view_frame.setLevels(lo, hi)

    # ------------------------------------------------------------------
    # SEGMENTATION OVERLAY
    # ------------------------------------------------------------------
    # Fallback colours for grayscale seg maps (up to 20; wraps for more).
    _SEG_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 120, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 64, 64), (64, 255, 64), (64, 64, 255),
        (200, 200, 0), (200, 0, 200), (0, 200, 200), (255, 160, 100),
        (100, 160, 255), (160, 255, 100), (220, 100, 160), (100, 220, 160),
    ]

    def _update_seg_overlay(self):
        """Render the segmentation mask for the current frame as a coloured RGBA overlay."""
        if self.seg_data is None or not self._seg_visible:
            self._seg_overlay.clear()
            return

        idx = self._current_frame_idx
        if idx >= self.seg_data.num_frames:
            self._seg_overlay.clear()
            return

        mask = self.seg_data.get_mask(idx)  # (H, W) int32, values = instance IDs

        # Resize mask if it doesn't match the video dimensions
        if self.video_data and (mask.shape[0] != self.video_data.height or
                                mask.shape[1] != self.video_data.width):
            import cv2
            mask = cv2.resize(mask, (self.video_data.width, self.video_data.height),
                              interpolation=cv2.INTER_NEAREST)

        alpha = self.slider_seg_opacity.value() / 100.0
        alpha_byte = int(alpha * 255)

        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        ids = np.unique(mask)
        ids = ids[ids != 0]

        # Check if status-color mode is active via the controller
        ctrl = getattr(self, '_controller', None)
        status_mode = ctrl is not None and ctrl._label_color_mode

        if status_mode and ctrl is not None:
            # Build lookup: instance_id → annotation state for this frame
            anno_state = {}  # instance_id → 'locked' | 'selected' | 'unlocked'
            for anno in ctrl.annotations:
                if anno.frame_idx == idx and anno.instance_id is not None:
                    if anno.is_locked:
                        state = 'locked'
                    elif anno.is_selected:
                        state = 'selected'
                    else:
                        state = 'unlocked'
                    anno_state[anno.instance_id] = state

            status_colors = {
                'locked':   (0, 200, 0),      # green
                'selected': (255, 255, 0),     # yellow
                'unlocked': (255, 50, 50),     # red
            }
            for iid in ids:
                iid_int = int(iid)
                state = anno_state.get(iid_int, 'unlocked')
                color = status_colors[state]
                region = mask == iid
                rgba[region, 0] = color[0]
                rgba[region, 1] = color[1]
                rgba[region, 2] = color[2]
                rgba[region, 3] = alpha_byte
        else:
            # Use original colours from the seg data
            inst_colors = getattr(self.seg_data, 'instance_colors', {})
            for iid in ids:
                iid_int = int(iid)
                if iid_int in inst_colors:
                    color = inst_colors[iid_int]
                else:
                    color = self._SEG_COLORS[(iid_int - 1) % len(self._SEG_COLORS)]
                region = mask == iid
                rgba[region, 0] = color[0]
                rgba[region, 1] = color[1]
                rgba[region, 2] = color[2]
                rgba[region, 3] = alpha_byte

        self._seg_overlay.setImage(rgba)

    def set_seg_visible(self, visible):
        """Toggle segmentation overlay visibility."""
        self._seg_visible = visible
        if visible:
            self._update_seg_overlay()
        else:
            self._seg_overlay.clear()
