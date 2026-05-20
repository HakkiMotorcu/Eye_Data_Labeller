from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QGroupBox, QListWidget,
                             QFileDialog, QMessageBox, QSlider, QScrollArea,
                             QSizePolicy, QSpinBox, QButtonGroup)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import os


class MainWindow(QMainWindow):
    def __init__(self, video_data=None):
        super().__init__()
        self.resize(1300, 800)
        self.video_data = video_data
        self.seg_data = None
        self._current_file = None
        self._current_frame_idx = 0

        pg.setConfigOptions(imageAxisOrder='row-major')
        self._setup_ui()
        if self.video_data:
            self.load_video()
        self._update_title()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top area: frame view + right panel ---
        top_layout = QHBoxLayout()

        # --- Left: Frame view ---
        frame_container = QWidget()
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

        top_layout.addWidget(frame_container, stretch=1)

        # --- Right: Scrollable Panels ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(350)
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

        # Add / Delete row
        add_del_row = QHBoxLayout()
        add_del_row.setSpacing(4)
        self.btn_add = QPushButton("Add  [A]")
        self.btn_add.setToolTip("Add a new annotation (A)")
        self.btn_add_vein = QPushButton("Vein [V]")
        self.btn_add_vein.setToolTip("Add a vein (paint-only, no bbox) (V)")
        self.btn_add_vein.setStyleSheet("color: #9370db;")
        self.btn_delete = QPushButton("Del [Del]")
        self.btn_delete.setToolTip("Delete the selected annotation")
        self.btn_delete.setStyleSheet("color: red;")
        add_del_row.addWidget(self.btn_add)
        add_del_row.addWidget(self.btn_add_vein)
        add_del_row.addWidget(self.btn_delete)
        list_layout.addLayout(add_del_row)

        # Rename / Fit row
        rename_fit_row = QHBoxLayout()
        rename_fit_row.setSpacing(4)
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setToolTip("Rename selected annotation (double-click list item)")
        self.btn_fit_bbox = QPushButton("Fit BBox")
        self.btn_fit_bbox.setToolTip("Fit bbox to actual seg pixels")
        rename_fit_row.addWidget(self.btn_rename)
        rename_fit_row.addWidget(self.btn_fit_bbox)
        list_layout.addLayout(rename_fit_row)

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
        self.btn_toggle_shape = QPushButton("BBox [O]")
        self.btn_toggle_shape.setCheckable(True)
        self.btn_toggle_shape.setToolTip("Toggle between BBox and Oval mode (O)")
        self.btn_hide_locked = QPushButton("Hide Lkd [H]")
        self.btn_hide_locked.setCheckable(True)
        self.btn_hide_locked.setToolTip("Toggle visibility of locked annotations (H)")
        toggle_row.addWidget(self.btn_toggle_shape)
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

        # 3. File I/O
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

        # SAM segmentation row
        sam_row = QHBoxLayout()
        sam_row.setSpacing(4)
        self.btn_run_sam = QPushButton("Run SAM")
        self.btn_run_sam.setToolTip("Run Segment Anything Model on current frame")
        self.btn_run_sam.setStyleSheet("color: #264653; font-weight: bold;")
        sam_row.addWidget(self.btn_run_sam)
        io_layout.addLayout(sam_row)

        # Export options checkboxes
        from PyQt6.QtWidgets import QCheckBox
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
        right_panel.addWidget(display_group)    # View
        right_panel.addWidget(io_group)         # I/O (Phase 2 will rework)
        right_panel.addStretch(1)

        scroll_area.setWidget(right_panel_widget)
        top_layout.addWidget(scroll_area)

        main_layout.addLayout(top_layout, stretch=1)

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

        self._init_level_sliders()
        self.display_frame(0, auto_range=True)

    def display_frame(self, idx, auto_range=False):
        """Display a specific frame in the view."""
        if not self.video_data:
            return
        idx = max(0, min(idx, self.video_data.num_frames - 1))
        self._current_frame_idx = idx

        frame = self.video_data.get_frame(idx)
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
