# Using Eye Data Labeller

## Workflow

1. **Launch** — double-click the Desktop launcher (or `python main.py`).
2. **Pick a file** — first dialog asks for a TIFF or AVI image stack.
3. **Annotate** — click on a frame to place a cell, drag corners to
   resize the bbox. Add vessels / capillaries with their dedicated
   buttons (or `V` / `C`).
4. **Segment** — switch to **paint** mode (`D`) to draw inside the
   bbox, or use **SAM Box** (`B`) to auto-fill the box from the
   fine-tuned SAM model.
5. **Lock + advance** — when an annotation is final, `Ctrl+L` locks
   it (read-only, won't be deleted) and moves to the next.
6. **Navigate frames** — `→` / `←` for one frame at a time,
   `Home` / `End` for first / last.
7. **Save** — `Ctrl+S` writes the segmentation map. Auto-save runs in
   the background every 30 sec (configurable in I/O Settings).

## Keyboard shortcuts

### Annotations

| Key | Action |
| --- | --- |
| `A` | Add a new annotation (cell, at last-used size) |
| `V` | Add a vessel |
| `C` | Add a capillary |
| `Delete` / `Backspace` | Delete selected annotation |
| `N` | Select next annotation |
| `P` | Select previous annotation |
| `L` | Lock selected (read-only) |
| `U` | Unlock selected |
| `Ctrl+L` | Lock selected AND advance to next |
| `H` | Toggle hide for locked annotations |

### Size presets

| Key | Action |
| --- | --- |
| `1` – `4` | Apply size preset 1 / 2 / 3 / 4 to selected |
| `0` | Capture current size as a preset |
| `T` | Apply last-used size |

### Segmentation modes

| Key | Action |
| --- | --- |
| `Escape` | Select mode (default — click to pick) |
| `D` | Paint mode (draw inside bbox) |
| `E` | Erase mode |
| `Shift+E` | Clear seg mask of selected |
| `F` | Fill the entire bbox |
| `B` | **SAM Box prompt** — auto-segment the bbox |
| `X` | Toggle force-paint (paint outside bbox) |
| `Ctrl+P` | Propagate vein mask across frames |

### Frame navigation

| Key | Action |
| --- | --- |
| `←` / `→` | Previous / next frame |
| `Home` | First frame |
| `End` | Last frame |

### View

| Key | Action |
| --- | --- |
| `R` | Reset zoom |

### File

| Key | Action |
| --- | --- |
| `Ctrl+S` | Save segmentation map |
| `Ctrl+I` | Import / load annotations |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

## Toolbar (left panel)

Most shortcuts above have a matching button in the toolbar. Hover
any button for a tooltip showing the action + its shortcut.

| Group | What's in it |
| --- | --- |
| **Annotations** | Add / delete / rename, lock & unlock (single or all), hide locked, label colors |
| **Filters** | Show All / Cells / Vessels / Capillaries |
| **Modes** | Select / Paint / Erase, Fill bbox, Force paint, Propagate mask |
| **Frames** | First / Prev / Next / Last, frame slider |
| **File** | Save seg, Load project, Import, I/O settings |
| **Image** | Auto-levels (contrast), fit bbox to content |

## I/O Settings (gear icon)

Open via the **I/O settings** button. Configures:

- **Output folder** — where seg maps + project JSON get saved.
  Three modes: subfolder of input, custom prefix, or fully custom
  path.
- **Auto-save** — interval, minimum flush interval for the on-disk
  mask, on/off.
- **SAM-HeLa checkpoint** — point at a local `best.pt` (Browse) OR
  paste a download URL. See `INSTALL.md` for details.

Settings persist across launches (stored via Qt's `QSettings`).

## Tips

- **First SAM Box on a frame is slow** (~3-5 sec on MPS, longer on
  CPU). The app pre-computes embeddings in the background, so
  subsequent boxes on the SAME frame are near-instant.
- **Force paint (`X`)** lets you paint outside the bbox. Useful for
  fixing SAM's spillover. Toggle it off when done to avoid surprises.
- **Locked annotations skip during `N` / `P`** — locking is your
  way of saying "done, leave alone."
- **Drag a bbox corner** to resize. Drag the bbox center to move.
- **Right-click an annotation** in the list for rename / delete /
  copy-to-next-frame.
- **Ctrl+L (lock-and-advance)** is the main rhythm during a session.
  Annotate → SAM Box → tweak → `Ctrl+L` → repeat.

## Where files live

| Thing | Path |
| --- | --- |
| Segmentation maps | Configured in I/O Settings — defaults to a subfolder of the input file's folder |
| Project JSON | Same folder as the seg maps |
| Auto-save scratch | In the output folder, prefixed with `.autosave_` |
| SAM-HeLa checkpoint | `models/checkpoints/sam_hela/best.pt` (or wherever you pointed) |
| SAM embedding cache | `~/.cache/eye_labeller/sam_embeddings/` |
| App settings | Qt `QSettings` — `~/Library/Preferences/com.EyeDataLabeller.*.plist` on macOS |

## Troubleshooting

- **`No SAM-HeLa checkpoint`** — drop `best.pt` at
  `models/checkpoints/sam_hela/best.pt` or set it via I/O Settings.
  See `INSTALL.md`.
- **SAM is slow** — confirm device with
  `python -c "from core.device import describe_device; print(describe_device())"`
  in the activated env. Should print `cuda (...)` or `mps (...)`.
- **App opens but no main window** — a file picker dialog opens
  first; check Mission Control / Cmd-Tab if it's hidden.
- **Anything else** — `python main.py --debug` enables verbose
  logging (sets `EYE_LABELLER_DEBUG=1`).
