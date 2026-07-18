---
name: ci-bundle-debugging
description: Debug CI/PyInstaller bundle failures in this repo (selftest crashes, hangs, silent SAM death, DLL/dylib symbol errors). Use when a Build-standalone-bundles job fails, when the bundled app behaves differently from the source tree, or before touching environment.yml / eye_labeller.spec / build.yml.
---

# CI bundle debugging — playbook

Everything here was learned the hard way (July 2026, PR #3 + deps/slim-micro-sam).
Read this before "fixing" a CI packaging failure — most of these bugs pattern-match
to a wrong cause first.

## The one root cause to remember

Almost every bundle failure in this repo has been a **native-library version
mismatch from the conda+pip hybrid env** — two builds of the same lib (Qt, ICU,
libtiff, libdeflate, libjpeg) land in the env, and the bundle (or Windows PATH)
picks the wrong one. The biggest single source was conda-forge `micro_sam`
hard-depending on **napari**, which dragged a second GUI stack (napari, vispy,
superqt, qtpy, PyQt5 + qt-main, qt6-main, conda ICU) into an app that never
imports it. That is why **micro-sam is pip-installed `--no-deps`** (see
environment.yml's micro-sam note) and must NEVER go back into the conda deps or
a plain pip section — upstream's pip metadata hard-pins napari too.

## Symptom → cause map (all previously solved — don't re-derive)

| Symptom | Real cause | Fix lives in |
|---|---|---|
| macOS bundle aborts, exit 134, "Could not find Qt platform plugin offscreen" | PyInstaller's PyQt6 hook doesn't collect `qoffscreen` | spec: explicit collect. MUST locate via bare `import PyQt6` package path — importing `PyQt6.QtCore` inside the spec **fails on Windows mid-build** and silently skips the block |
| Windows bundle **hangs** (no output, eats the 15-min cap) at startup | Same missing-plugin error, but on Windows Qt pops a **modal dialog** instead of aborting. Secondary: missing `multiprocessing.freeze_support()` (torch/kornia spawn workers that re-launch the frozen exe into the GUI) | spec (plugin) + top of main.py (freeze_support) |
| App runs but **SAM silently dead** (no crash!) | `core/sam_service.py` catches ImportError by design. Excluding `matplotlib` from the bundle kills a deep micro_sam/torch_em import → invisible failure. Grep bundle-selftest logs for `micro_sam import failed` — a green job does NOT mean SAM works | spec excludes: matplotlib excluded on **Linux only** |
| Shipping matplotlib crashes **Linux** bundle at `pg.ImageView()` | pyqtgraph's ColorMapMenu eagerly imports matplotlib when present; its bundled native stack has an undefined symbol in frozen Linux (was libtiff `jpeg12_write_raw_data`, then libraqm `hb_ft_font_get_ft_face`) | spec: Linux-only matplotlib exclude |
| Windows source tree: `DLL load failed while importing QtCore: The specified procedure could not be found` | conda ICU in `<env>\Library\bin` (on PATH) lacks the **unversioned** `UCNV_TO/FROM_U_CALLBACK_SUBSTITUTE` symbols; PyQt6's Qt6Core is built against **System32's ICU**, which has them. NOT the VC runtime (conda's msvcp140 is newer = compatible). NOT Qt-vs-Qt (absolute-path load of pip's Qt6Core also failed) | main.py preloads `System32\icuuc.dll`/`icuin.dll` before first Qt import. Keep even though napari is gone — icu can persist via boost/vigra |
| Linux bundle: `libOpenEXRCore.so.33: undefined symbol: libdeflate_alloc_compressor_ex` → SAM dead on Linux | PyInstaller shipped OpenEXR (linked by conda **vigra**) without the env's libdeflate, so the loader fell back to Ubuntu's system libdeflate 1.10 (predates the `_ex` API). Spec now force-bundles the env's libdeflate | spec (Linux block). NOTE: **Linux was dropped from the CI matrix 2026-07-18** (collaborators are Win+mac only) — the fix and this row are kept for a possible future re-enable; Tier B Linux (conda env) still works |
| cv2 "recursion is detected", pillow `jpeg12` crash, numpy `_ctrsyl3_` on mac | conda/pip duplicate flavors or wrong BLAS | build.yml has dedicated steps; read their comments before touching |

## Method — evidence first (two blind fixes failed before this worked)

1. **Never guess twice.** After one failed fix, add a **temporary CI diagnostic
   step** that dumps ground truth, and only then write the real fix. Validate the
   fix mechanism *inside the diagnostic step in the same run* before committing it.
2. **`pefile` settles DLL questions** (which DLL is bad, which symbol is missing):
   list a binary's imports, intersect with `Library\bin` contents for shadow
   suspects; dump a candidate DLL's exports and check for the exact wanted symbols.
3. **Windowed exe on Windows loses stdout** (block-buffered, discarded on kill).
   Run the bundle selftest with `PYTHONUNBUFFERED=1` and set
   `EYE_LABELLER_FAULT_TIMEOUT=<sec>` — main.py's faulthandler watchdog dumps
   every thread's stack and aborts, turning a silent hang into a traceback.
4. **Read the app's own log** when stdout is gone: `%LOCALAPPDATA%\EyeDataLabeller\logs\session_*.log`
   (Windows) / `~/Library/Application Support/EyeDataLabeller/logs` (mac).
5. **Fetch logs of a finished job while the run is still going**:
   `gh api repos/<owner>/<repo>/actions/jobs/<job-id>/logs`
   (`gh run view --log` refuses until the whole run completes). Windows finishes
   first (~9–12 min) — don't wait for mac/Linux to read it.
6. **Dispatch + watch**: `gh workflow run build.yml --ref <branch>` then
   `gh run watch <run-id> --exit-status` in the background. A full 3-OS run is
   ~15–35 min; budget several iterations for anything Windows-DLL-shaped.
7. **A green job ≠ working SAM.** Always grep the bundle-selftest output for
   `micro_sam import failed` before declaring victory (sam_service swallows it).

## Guardrails already in CI (don't remove)

- The **"Install micro_sam (slim, --no-deps)"** step asserts
  napari/magicgui/superqt/vispy are absent and imports the compute surface —
  if a future env change sneaks the GUI stack back in, that step fails with a
  clear message. If you see it fail, fix the dependency that pulled napari in;
  do not delete the assert.
- Source-tree selftest runs **before** the bundle selftest on purpose: source
  failure = env problem, bundle-only failure = packaging problem. This
  distinction is the first branch of every diagnosis.
- The Windows bundle selftest keeps `PYTHONUNBUFFERED=1` +
  `EYE_LABELLER_FAULT_TIMEOUT=300` so any future startup hang self-diagnoses.

## When bumping micro-sam

1. Change the tag in: environment.yml header note, the CI slim-install step, and
   all three `deploy/install_*` scripts (grep for `micro-sam.git@`).
2. Check the new tag's compute modules for **new module-level imports**
   (util.py, instance_segmentation.py, automatic_segmentation.py,
   multi_dimensional_segmentation.py, prompt_based_segmentation.py) and add any
   new deps to environment.yml explicitly — `--no-deps` means nothing arrives
   automatically. (Example: master already imports `bioimage_cpp`, which 1.7.7
   does not.)
3. Let the CI assert step catch anything missed.
