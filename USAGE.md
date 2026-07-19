# Using Eye Data Labeller

## Workflow

1. **Launch** — double-click the Desktop launcher (or `python main.py`).
   The app opens on a **landing page**: an Open button, a drop target,
   and your recent files with work-status glyphs — **✓ complete**,
   **● in progress**, untouched files sorted to the bottom. A single
   click opens a file; right-click removes entries. The gear opens
   Settings. Editor shortcuts and file-dependent menus stay disabled
   until a file is open.
2. **Open a file** — click **Open image / video…** (or drop a
   TIFF/video onto the window, click a recent, or use the **Files
   sidebar**'s **Next ▶**). You never need to relaunch to switch
   files: `Ctrl+O`, drag-and-drop, **File → Open Recent**, or the
   sidebar all open a new stack in place, and **File → Close**
   (`Ctrl+W`) takes you back to the landing page. When you leave a
   stack with live work, a dialog asks how to record it — **Save &
   mark complete** or **Save & mark in progress** — and that's what
   the ✓ / ● glyphs everywhere show.
3. **Annotate** — click on a frame to place a cell, drag corners to
   resize the bbox. Add vessels / capillaries with their dedicated
   buttons (or `V` / `C`).
4. **Segment** — switch to **paint** mode (`D`) to draw inside the
   bbox, or use **SAM Box** (`B`). SAM's mask appears first as a cyan
   **preview** — press `Enter` (or `B` again) to accept it, `Esc` to
   throw it away. Nothing touches your data until you accept.
5. **Lock + advance** — when an annotation is final, `Ctrl+L` locks
   it (read-only, won't be deleted) and moves to the next.
6. **Track cells** — with a segmentation on multiple frames, the
   Tracking panel's **Run Tracker** coalesces the same cell across
   frames into one identity (shared color/name). **Track lengths…**
   then shows a table: each track's length (how many frames the cell
   spans), its frame range, and any gaps — sorted longest first, with
   a summary line (count, longest, median, single-frame tracks). The
   per-frame `track_length` also lands in the Export Bundle's CSV.
   **Double-click a track** to jump to its first frame and zoom to the
   cell; the footer's **Re-run tracker** drops tracks shorter than the
   chosen length (single-frame tracks are the usual tracking-error
   signal). Tracker options — mode, gap-closing, min length — live in
   the Tracking panel.
7. **Navigate frames** — `→` / `←` for one frame at a time,
   `Home` / `End` for first / last, `Ctrl+→` / `Ctrl+←` to jump to
   the next / previous frame with no annotations. The tick bar above
   the timeline shows which frames carry work.
8. **Save** — `Ctrl+S` writes the project (mask TIFs + Meta.json +
   project.json) atomically. The **status bar** shows the state at all
   times, Word-style: *● Unsaved changes* → *✓ Saved 2 min ago*.
   Auto-save runs in the background every 30 sec (configurable in
   Settings). Two safety nets: every save keeps the previous file as
   `<file>.bak`, and the first save of each editing session snapshots
   the folder's existing masks into `backup/session-<timestamp>/` (so
   resuming and saving twice can't destroy what you resumed from). If
   you were working boxes-only and never saved masks, reopening the
   video offers to **restore** the autosaved annotation snapshot.

The **toolbar** (top-left) has the sidebar toggle (`Ctrl+B`), Open,
and Save; the **File** menu holds Open / Open Recent / Close / Save /
Load Project Folder / Import Annotations / Load Single-Class TIF /
Export Bundle. There is one dialog for leaving a session (switch file,
Close, or quit): **Save & mark complete / in progress / Discard /
Cancel**.

## Working through many files — the Files sidebar

**View → Files Sidebar** (on by default) is a file explorer rooted at
a folder of your choosing (**Choose folder…**); only supported stacks
are shown. Each file carries its work status at the right edge:

- **✓** (green) — you marked it complete when leaving it
- **●** (yellow) — session artifacts exist without that mark
- nothing — untouched

Double-click opens a file (the leave dialog guards your current
session). **Next ▶** opens the first top-level file not marked ✓ —
a folder of 30 stacks becomes a next-next-next session instead of 30
launches. Right-click a file to open it or correct its status
(*Mark ✓ complete / Mark ● in progress / Clear status*). Statuses
live in each stack's output folder (`project.json`), so they travel
with the data.

## Review mode

**View → Review Mode** (`Ctrl+R`) walks every *unlocked* annotation
in frame order: each one is selected and zoomed to, the status bar
shows `REVIEW k/N`. Press `Space` to accept it (locks it — undoable)
and jump to the next; fix things with any normal tool in between;
`Esc` exits. When the counter completes, everything is locked and the
stack is review-clean.

## Checking against the previous frame

**View → Onion Skin** (`O`) draws faint outlines of the previous
frame's masks under the current frame — tracking drift and missed
cells show up immediately while you scrub.

## Sharing results — Export Bundle

**File → Export Bundle** writes `<out folder>/export/` containing
mask TIF snapshots, `Overlay.mp4` (annotations burned into the video
— anyone can scrub it without installing anything), and `Summary.csv`
(one row per instance per frame: class, name, pixel area, bbox).

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
| `B` | **SAM Box prompt** — preview the mask (again/`Enter` accepts, `Esc` discards) |
| `X` | Toggle force-paint (paint outside bbox) |
| `S` | Toggle segmentation overlay |
| `Ctrl+wheel` | Brush size (in paint / erase modes) |
| `Ctrl+P` | Propagate vein mask across frames |

### Frame navigation

| Key | Action |
| --- | --- |
| `←` / `→` | Previous / next frame |
| `Home` | First frame |
| `End` | Last frame |
| `Ctrl+←` / `Ctrl+→` | Previous / next **unannotated** frame |

### View

| Key | Action |
| --- | --- |
| `R` | Reset zoom |
| `Z` | Zoom to selected annotation |
| `O` | Onion skin (previous frame's outlines) |

### Review mode

| Key | Action |
| --- | --- |
| `Ctrl+R` | Enter / leave review mode |
| `Space` | Accept (lock) current + advance |
| `Esc` | Exit review mode |

### File

| Key | Action |
| --- | --- |
| `Ctrl+O` | Open image / video in place |
| `Ctrl+B` | Show / hide the Files sidebar |
| `Ctrl+W` | Close file — back to the landing page |
| `Ctrl+S` | Save segmentation map |
| `Ctrl+I` | Import / load annotations |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` / `Ctrl+Y` | Redo |
| `Ctrl+Q` | Quit |
| `F1` | Keyboard shortcut reference (in-app) |

## Toolbar (left panel)

Most shortcuts above have a matching button in the toolbar. Hover
any button for a tooltip showing the action + its shortcut.

| Group | What's in it |
| --- | --- |
| **Annotations** | Add / delete / rename, lock & unlock (single or all), hide locked, label colors |
| **Filters** | Show All / Cells / Vessels / Capillaries |
| **Modes** | Select / Paint / Erase, Fill bbox, Force paint, Propagate mask |
| **Frames** | First / Prev / Next / Last, frame slider |
| **File** | Save seg, Load project, Import, Settings |
| **Image** | Auto-levels (contrast), fit bbox to content |

## Settings (gear icon / File → Settings)

One panel, four pages (category list on the left):

- **Output & Autosave** — where seg maps + project JSON get saved
  (subfolder of input, custom prefix, or fully custom path) and the
  auto-save mode/intervals.
- **SAM Model** — point at a local `best.pt` (Browse) OR paste a
  download URL. See `INSTALL.md` for details. The **Model** menu in
  the menu bar (available on the landing page too) is the quick path:
  it shows the current model and *Choose checkpoint file…* asks once
  and remembers. The app never demands a model — without one, SAM
  assist is simply off (status line says so) and manual annotation
  works normally.
- **Detection** — SAM auto-segmentation tuning: custom quality /
  stability thresholds (stricter = fewer, cleaner cells) and a
  min/max pixel-area size filter that drops specks and merged blobs
  before they become annotations. Defaults are the model's own;
  applies to Auto-segment, not SAM Box.
- **Annotation** — quality flags (below), and future annotation
  defaults.
- **Debugging** — detailed logging toggle + log folder.

Settings persist across launches (stored via Qt's `QSettings`).

## Quality flags (⚠ in the annotation list)

Suspicious cells get a `⚠` next to their class in the list — hover
the row's Class column to see why. Three deterministic geometry
checks (no model involved), each catching a known failure mode:

- **Mask touches its bbox edge** — SAM probably spilled past the cell.
- **Mask split into disconnected pieces** — one "cell", two blobs.
- **Area far from the frame's median cell** (>4× or <¼×) — probable
  merge or speck.

The flag only says *look at me* — you decide. Enable/disable the
whole feature or individual checks in **Settings → Annotation**.

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
| Segmentation maps | Configured in Settings — defaults to a subfolder of the input file's folder |
| Project JSON | Same folder as the seg maps |
| Auto-save scratch | In the output folder, prefixed with `.autosave_` |
| SAM-HeLa checkpoint | `models/checkpoints/sam_hela/best.pt` (or wherever you pointed) |
| SAM embedding cache | `~/.cache/eye_labeller/sam_embeddings/` |
| App settings | Qt `QSettings` — `~/Library/Preferences/com.EyeDataLabeller.*.plist` on macOS |

## Troubleshooting

- **`No SAM-HeLa checkpoint`** — **Model → Choose checkpoint file…**
  (asks once, remembered), or drop `best.pt` at
  `models/checkpoints/sam_hela/best.pt`. See `INSTALL.md`.
- **SAM is slow** — confirm device with
  `python -c "from core.device import describe_device; print(describe_device())"`
  in the activated env. Should print `cuda (...)` or `mps (...)`.
- **App opens but no main window** — the window (landing page) can
  open behind other apps; check Mission Control / Cmd-Tab.
- **Anything else** — `python main.py --debug` enables verbose
  logging (sets `EYE_LABELLER_DEBUG=1`).
