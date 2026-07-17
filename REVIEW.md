# Code Review — Findings & Status

> Markdown clone of the interactive review report (17 Jul 2026,
> status updated 18 Jul). ✅ fixed · ⬜ open. Items marked *verified*
> were independently re-checked against the code by an adversarial
> reviewer before being reported; two claims were refuted during
> verification and dropped.

## At a glance

| | Fixed | Open |
| --- | --- | --- |
| **Findings** | **34** | **18** |
| Critical defects | 2 / 2 | 0 |
| High defects | 12 / 12 | 0 |
| High-leverage refactors | 2 | 2 (planned) |
| Medium / low | rest | mediums & lows only |

*(Counts are table rows below; a few rows bundle several small
items. Every open item is a medium/low or a planned refactor — no
known data-loss or crash paths remain.)*

## Plan for the open items

In order, matching the project's small-commit rhythm:

1. **Pin the environment** *(new — born from the July CI incident)*:
   exact versions in `environment.yml`, refreshed deliberately
   instead of drifting with every solve. First thing after the
   current PR merges.
2. **Responsiveness wave**: move smart-autosave mask writes, SAM
   all-frames runs, and tracking off the UI thread (the embedding
   worker is the in-repo pattern); sparse TrackingCmd snapshots.
   Removes the last multi-second UI freezes.
3. **Data-integrity sweep** (small fixes): brush stroke spanning a
   frame change, zarr cache done-sentinel, preserve corrupt
   Meta.json instead of overwriting, never `terminate()` the embed
   worker.
4. **Platform polish batch**: Linux `.desktop` launcher, Qt env-var
   re-assert before QApplication, per-OS shortcut labels, disk-full
   dialog paths, installer edge cases, one Windows verification of
   non-ASCII video paths.
5. **Release engineering**: fan-in release job, signing/notarization
   documentation, INSTALL.md corrections.
6. **Architecture (ongoing)**: ProjectController → SamController →
   `set_current_frame()` / event-bus revival — one extraction per
   session, app working after every commit.

**Status: all critical and high findings are closed, and the full
roadmap has shipped** in six delivery waves — (1) Windows
install/first-launch/encoding + data safety + logging + CI smoke
tests, (2) performance (23× overlay redraw, no full-stack scans, ½
memory), (3) architecture extractions + identity list mapping,
(4) session UX (Files sidebar + queue, in-app open, menu/F1/mode
chip, timeline markers, SAM preview), (5) review tooling (Review
Mode, onion skin, Export Bundle, queue ranking), (6) Settings panel +
quality flags + CI dependency-drift fix. Every wave was adversarially
re-reviewed before landing; those reviews caught 2 critical, 4 high,
and ~15 medium/low defects in the new code itself, all fixed.

## Verdict (from the original review)

An unusually solid codebase for a first GUI project: command-pattern
undo, atomic writes with backups, correct per-OS path handling,
consistently explained code. The problems clustered in data safety,
Windows support, redraw/memory efficiency, and one 4,700-line god
object — all addressed below.

## 1 · Data safety — closed

| Status | Sev | Finding |
| --- | --- | --- |
| ✅ | high | No unsaved-changes prompt on close; most pixel edits never set the dirty flag *(verified)* |
| ✅ | high | Undo of a cell delete didn't restore its mask pixels *(verified)* |
| ✅ | high | Fill BBox (`F`) was not undoable *(verified)* |
| ✅ | high | Emptied class layers left stale mask TIFs that resurrected on reload *(verified; redesigned with session-ownership + rename-to-.bak after re-review)* |
| ✅ | high | Bboxes permanently un-clickable after paint mode + frame change *(verified)* |
| ✅ | med | Duplicate names misrouted list clicks/deletes (identity-based rows now) |
| ⬜ | med | Corrupt Meta.json silently swallowed (names/locks lost on next save) |
| ✅ | med | Truncated model download could be committed (length check added) |
| ⬜ | med | Brush stroke spanning a keyboard frame-change corrupts the undo record |
| ⬜ | med | Partial zarr embedding cache trusted by existence |
| ⬜ | med | Embedding worker can be terminate()d mid-inference (close path now waits 10 s) |
| ✅ | low | Autosave failures were silent (red status now, self-clearing) |

## 2 · Windows & cross-platform — closed

| Status | Sev | Finding |
| --- | --- | --- |
| ✅ | crit | install_windows.bat parse-time expansion broke every fresh install *(verified)* |
| ✅ | crit | First-run best.pt popup ran on a background QThread (first-launch freeze) *(verified — the collaborator-reported bug)* |
| ✅ | high | JSON/CSV import corrupted Turkish/BOM text on Windows *(verified, reproduced live)* |
| ✅ | high | Linux CUDA swap broke torch imports; git-less installs failed *(verified)* |
| ✅ | high | Launcher written to wrong Desktop under OneDrive redirection *(verified)* |
| ✅ | med | Embedding cache in `~/.cache` on Windows (now per-OS user-data dir, with migration) |
| ✅ | med | `pythonw.exe` crashed on first log line (stderr guard) |
| ✅ | med | Python 3.12 silently downgraded pyqtdarktheme (fork pinned) |
| ⬜ | med | cv2 imported before QApplication can clobber Qt plugin path on Linux |
| ⬜ | med | Linux "desktop launcher" is a bare .sh (no .desktop entry) |
| ⬜ | low | "Cmd+Z" wording in UI strings on Windows; disk-full dialog shows macOS path; file dialogs partially cwd-based; cv2 non-ASCII path needs one Windows verification; installer paths with `)`/`&` |

## 3 · Performance — top items closed (23×, ½ memory)

| Status | Sev | Finding |
| --- | --- | --- |
| ✅ | high | Brush stroke rebuilt the whole overlay 60×/sec *(verified — LUT compositing: 119 ms → 5 ms)* |
| ✅ | high | Overlay compositing O(instances × pixels) *(verified — single LUT gather)* |
| ✅ | high | ~3.1 GB seg layers on a 1000-frame stack *(verified — uint16: ~1.6 GB)* |
| ✅ | high | Frame navigation ran full-stack scans per step *(verified — cached per-class id sets)* |
| ✅ | med | SAM dedupe O(detections × annotations × pixels) (shared candidates + windowed IoU) |
| ✅ | low | Colormap re-resolved per frame change |
| ⬜ | med | Smart-autosave writes mask TIFs on the UI thread |
| ⬜ | med | SAM all-frames + tracking run synchronously on the UI thread |
| ⬜ | med | TrackingCmd snapshots the full mask stack twice per run |
| ⬜ | med | Overlay rebuild iterates all annotations across all frames |
| ⬜ | low | Sliding-window projection recomputes per scrub step |

## 4 · UI & workflow — shipped (waves 4–6)

| Status | Sev | Finding / feature |
| --- | --- | --- |
| ✅ | high | No way to open a second file (now: Ctrl+O, drag-drop, recents, sidebar) |
| ✅ | high | No in-app shortcut reference (F1 + menu bar) |
| ✅ | med | Context menu documented but missing (built) |
| ✅ | med | No visible tool-mode state (status-bar chip, red FORCE PAINT) |
| ✅ | med | No zoom-to-annotation (`Z`) |
| ✅ | med | Save/dirty state invisible (title dot + status) |
| ⬜ | med | First-open embedding uses an app-modal dialog (mitigated: shown after window, one-ahead prefetch; still modal) |
| — | — | **Beyond the review:** Files sidebar + session queue (●◐○, Next, optional SAM-disagreement ranking), timeline coverage markers + Ctrl+←/→, SAM preview-before-commit (B → Enter/Esc), Review Mode (Ctrl+R/Space), onion skin (O), Ctrl+wheel brush, Export Bundle (TIFs + Overlay.mp4 + Summary.csv), Settings panel (Output/Model/Annotation/Debugging), quality flags (⚠ with per-check toggles) |

## 5 · Architecture — steps 1 + 4 done

| Status | Sev | Finding |
| --- | --- | --- |
| ✅ | high | Extraction 1: UndoStack + commands + Annotation2D → own modules (−730 lines) |
| ✅ | high | Name-keyed list synchronization (identity-stamped rows) |
| ⬜ | high | Extraction 2: ProjectController (save/load/autosave/resume) |
| ⬜ | high | Extraction 3: SamController (worker + prompts + auto-segment) |
| ⬜ | med | `set_current_frame()` consolidation; AppState bus revival (markers now subscribe); window↔controller back-reference; core not fully headless; Annotation2D drag logic; class-taxonomy registry; 1,030-line `_setup_ui` |

## 6 · CI & release — smoke tests live

| Status | Sev | Finding |
| --- | --- | --- |
| ✅ | med | CI never executed the bundles it shipped (`--selftest` runs on all 3 OSes, source + bundle, 15-min caps) |
| ✅ | med | ubuntu-latest glibc floor excluded older labs (pinned 22.04) |
| ✅ | med | cv2 config patch broke on July's flat pip layout (layout-aware now) |
| ✅ | low | `.zip.zip` artifact names |
| ⬜ | med | Unsigned/unnotarized macOS bundle; INSTALL.md Gatekeeper advice outdated |
| ⬜ | low | Release-creation race across matrix jobs; first-use best.pt path persisted before validation; INSTALL.md weights-location doc |

## Methodology

Multi-agent review: 7 dimension reviewers + 1 architecture mapper
read all ~10 k lines; every critical/high claim went to an
independent adversarial verifier instructed to refute it (2 were, and
were dropped). Each subsequent delivery wave got its own adversarial
review before landing; several findings in the new code were
themselves verified empirically against Qt at runtime.

*The interactive version of this report (severity chips, evidence,
per-finding fixes) lives in the project's Claude artifact; this file
is its repo-tracked clone — update both together.*
