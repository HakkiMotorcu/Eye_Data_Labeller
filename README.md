# Eye Data Labeller

A PyQt6 desktop tool for annotating cells, vessels, and capillaries
in retinal microscopy stacks (TIFF / AVI), with one-click instance
segmentation powered by a fine-tuned [Segment Anything](https://segment-anything.com/)
model (SAM-HeLa).

Built for fast, repeatable annotation sessions on multi-frame data —
draw a bounding box, hit `B`, get a clean mask. Lock and advance.

---

## Quick start

```bash
git clone https://github.com/HakkiMotorcu/Eye_Data_Labeller.git
cd Eye_Data_Labeller

# macOS:
deploy/install_mac.command          # Finder → right-click → Open
# Linux:
bash deploy/install_linux.sh
# Windows (cmd.exe):
deploy\install_windows.bat
```

The installer drops a launcher on your Desktop. Double-click it.
The app opens on a landing page — open a stack from there (button,
drag-and-drop, recent files, or the session queue).

Full install details (GPU support, model weights, troubleshooting) →
[**INSTALL.md**](INSTALL.md).

Using the app (workflows, all keyboard shortcuts, tips) →
[**USAGE.md**](USAGE.md).

---

## Features

- **Multi-frame TIFF + AVI** input (works on full video stacks).
- **Three annotation classes** — cells, vessels, capillaries —
  each with its own colour, list filter, and lifecycle.
- **SAM Box** — one-click bounding-box → segmentation mask via a
  fine-tuned SAM-HeLa checkpoint. Pre-computed embeddings make
  subsequent boxes on the same frame near-instant.
- **Paint / erase brush** for manual edits on top of (or instead of)
  SAM output.
- **Lock + advance** workflow — `Ctrl+L` marks an annotation final
  and jumps to the next, so a session has a consistent rhythm.
- **Size presets** (`1`–`4` + capture `0`) for quickly applying a
  known cell / vessel size to a new annotation.
- **Auto-save** — periodic writes of the seg map + project JSON,
  configurable interval, atomic rename so a crash never leaves a
  half-written file.
- **GPU-aware** — CUDA on Win/Linux + NVIDIA, MPS on Apple Silicon,
  CPU fallback. Device auto-detected at startup; override with
  `EYE_LABELLER_DEVICE=cpu` / `cuda` / `mps`.

---

## Distribution

Two install paths, both work end-to-end on macOS / Windows / Linux:

| Path | Best for |
| --- | --- |
| **Tier B** — `git clone` + `deploy/install_*.{sh,bat,command}` | Labs that update often (`git pull` + re-run installer). Has GPU auto-detect. |
| **Tier A** — Download bundle from [Releases](https://github.com/HakkiMotorcu/Eye_Data_Labeller/releases) | Non-developer collaborators. No terminal, no conda. |

See [INSTALL.md](INSTALL.md) for the full comparison.

---

## Built with

- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) + [pyqtgraph](https://www.pyqtgraph.org/) — UI
- [micro-sam](https://github.com/computational-cell-analytics/micro-sam) + [Segment Anything](https://github.com/facebookresearch/segment-anything) — segmentation backbone
- [PyTorch](https://pytorch.org/) — model runtime (CUDA / MPS / CPU)
- [trackastra](https://github.com/weigertlab/trackastra) — cell tracking (Phase 4)
- [scikit-image](https://scikit-image.org/), [opencv](https://opencv.org/), [tifffile](https://github.com/cgohlke/tifffile) — image I/O & processing

---

## Project layout

```
Eye_Data_Labeller/
├── main.py                # Entry point
├── core/                  # Domain logic (no Qt deps)
│   ├── device.py            # cuda / mps / cpu picker
│   ├── sam_service.py       # micro-sam wrapper + embedding cache
│   ├── model_download.py    # SAM-HeLa checkpoint resolver
│   ├── app_paths.py         # bundled / user-data path helpers
│   ├── volume_data.py       # TIFF / AVI loaders
│   ├── project_io.py        # save / load project state
│   └── …
├── ui/                    # Qt widgets
│   ├── main_window.py       # Image view + panels + status bar
│   ├── landing_page.py      # Fileless home screen
│   ├── session_queue.py     # Work-list widget (landing + Files dock)
│   ├── files_panel.py       # Files dock: browser + queue
│   └── settings_dialog.py   # Settings (output, SAM model, …)
├── controllers/
│   └── tool_controller.py   # Wires UI events → core logic
├── deploy/                # Install scripts + helpers
├── packaging/             # PyInstaller spec for Tier A bundles
├── models/checkpoints/sam_hela/   # Drop best.pt here
├── INSTALL.md             # Setup guide
└── USAGE.md               # Shortcuts + workflow
```

---

## License

TBD — pick one before sharing publicly (recommend MIT or Apache 2.0
for permissive academic use).
