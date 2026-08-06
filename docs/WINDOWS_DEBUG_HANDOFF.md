# Windows Debug Handoff — "micro_sam is not installed"

**For:** a Claude Code session running on a Windows machine.
**Written:** 2026-08-06, from the macOS session that shipped v0.3.2.
**Goal:** find out why SAM does not enable on a collaborator's Windows
machine, and fix it.

Read this whole file before running anything. Section 5 is the actual
plan; sections 1–4 are the context you need to not waste the session
re-deriving what is already known.

---

## 1. The symptom

On one collaborator's Windows machine the app launches and annotates
fine, but SAM never works. The SAM panel reads **"model: micro_sam not
installed"**.

That string is not about the checkpoint. In the code there are two
different states and they are worded differently:

| App shows | Meaning | Source |
|---|---|---|
| `micro_sam not installed` | `import micro_sam` **failed** | `SAM_AVAILABLE = False` in `core/sam_service.py` |
| `checkpoint missing` | micro_sam is fine, the `.pt` file isn't there | registry entry with a dead path |

So this is a **Python import failure**, not a model-path or model-name
problem. The user already confirmed the built-in models (`vit_t`,
`vit_b`, …), which need no local checkpoint at all, fail the same way —
consistent with the import diagnosis.

The relevant code (`core/sam_service.py`, near the top):

```python
try:
    import micro_sam
    from micro_sam import util
    from micro_sam.automatic_segmentation import (...)
    SAM_AVAILABLE = True
except ImportError as _exc:
    log_error('sam_service', 'micro_sam import failed', exc=_exc)
    SAM_AVAILABLE = False
```

**The exception is already written to disk with a full traceback**, on
every machine, with no debug flag required (`log_error` always writes
the traceback to the log file; only stderr is gated by `--debug`).
Getting that traceback is the whole job.

---

## 2. What is already ruled out — do not redo this

| Checked | Evidence | Conclusion |
|---|---|---|
| Windows **source install** recipe | CI Windows job: `selftest: SAM vit_t loaded + box prompt OK (mask px=576)` | The recipe works on a clean Windows box |
| Windows **bundle** | CI asserts `SAM_AVAILABLE` inside the built bundle (PR #13) since v0.3.1 | Bundle import chain is healthy as built |
| Model naming (`best.pt` vs other names) | Two distinct UI strings, see §1; built-ins fail too | Not a naming problem |
| macOS end-to-end from zero | Fully isolated install + real segmentation, all pass | Not a cross-platform code bug |
| Our packaging recipe generally | v0.3.2 CI green on both OSes | Healthy |

The failure is **environment-specific to that machine**. CI is a clean,
GPU-less, git-preinstalled, no-antivirus, no-OneDrive, no-proxy runner.
The collaborator's machine is none of those things.

---

## 3. Ranked hypotheses

Ordered by likelihood × how badly CI fails to cover them. The top two
are **code paths CI never executes** — exactly the blind spot that hid
the Miniforge installer bug found on 2026-08-06 (see §7).

### H1 — CUDA PyTorch swap broke torch  ⚠️ strongest, source install
`deploy/install_windows.bat` step 3 detects an NVIDIA GPU and runs:

```bat
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

- **CI never runs this** — GitHub runners have no NVIDIA GPU, so the
  whole branch is skipped. The collaborator's annotation workstation
  almost certainly *does* have one.
- `--no-deps` + `--force-reinstall` can leave a torch/torchvision pair
  that mismatches each other or the pinned numpy, or a cu124 build the
  installed driver is too old for (needs driver ≥ 550; cu121 needs
  ≥ 525 — the header of the .bat documents this).
- A broken torch surfaces as `import micro_sam` failing, because
  micro_sam imports torch → **exactly our symptom**.
- Note the `|| (echo ... )` fallback in the .bat only prints a warning
  and **continues**, so a half-broken swap does not stop the install.

**Confirm:** in the env, `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
If that traceback is a DLL/symbol error, H1 is your answer.

### H2 — `git` is not installed → micro-sam never installed  ⚠️ source install
The installer installs micro-sam from a **git URL**:

```bat
pip install --no-deps "git+https://github.com/computational-cell-analytics/micro-sam.git@v1.7.7" || goto :fail
```

`pip` shells out to `git.exe` for this. A fresh Windows machine
frequently has **no git at all** (macOS gets it with the Xcode CLT; CI
runners ship it preinstalled). `install_windows.bat` **never checks for
git** — I verified this. The step does `goto :fail`, so the install
*should* abort loudly, but a user who missed the red text (or ran the
app from the partially-built env anyway) ends up with an env that has
everything **except** micro_sam.

**Confirm:** `pip show micro-sam` in the env (empty = never installed),
and `git --version` at a plain prompt.

**Fix if confirmed** — drop the git dependency entirely by installing
the release tarball instead:
```bat
pip install --no-deps "https://github.com/computational-cell-analytics/micro-sam/archive/refs/tags/v1.7.7.tar.gz"
```
…plus an explicit up-front `where git` check with a clear message.

### H3 — Corporate network / proxy blocked the pip download
University-managed machines often block `github.com` over git protocol
or MITM TLS. Same end state as H2 (no micro_sam), different cause.
**Confirm:** the install console output, or `pip show micro-sam`.

### H4 — Old bundle
If they ran a bundle **older than v0.3.1**, it predates the
`SAM_AVAILABLE` assertion, so a broken bundle could have shipped
silently. **Confirm:** ask exactly which zip they downloaded, and check
the app's title/Help→About against v0.3.2.

### H5 — Antivirus / OneDrive interference with the bundle
Managed Windows machines quarantine unsigned `.exe`/`.dll` files inside
an extracted zip, or redirect Desktop to OneDrive so paths shift.
A missing DLL inside `_internal` breaks the torch import chain →
same symptom. **Confirm:** the log traceback naming a specific DLL.

### H6 — Wrong env activated
If the machine already had Anaconda, the launcher's
`call ...\Scripts\activate.bat eye-labeller` may activate a *different*
`eye-labeller`. **Confirm:** in the app log's startup banner, or
`python -c "import sys; print(sys.prefix)"`.

---

## 4. Where the evidence lives

**Log files (both install types):**
```
%LOCALAPPDATA%\EyeDataLabeller\logs\session_*.log
```
Open the **newest** one and search for `micro_sam import failed`. The
lines indented under it are the real exception and traceback — that one
block decides between H1/H2/H5/H6 immediately.

PowerShell one-liner to pull it:
```powershell
Get-ChildItem "$env:LOCALAPPDATA\EyeDataLabeller\logs\session_*.log" | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content | Select-String -Pattern "micro_sam import failed" -Context 0,25
```

**Source installs additionally allow:**
```bat
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" eye-labeller
python -c "import micro_sam"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip show micro-sam
python -c "import sys; print(sys.prefix)"
```
The first command prints the full traceback directly — fastest possible
answer.

**Bundle installs have no Python**, so the log file is the only route.
That is why the pending "Diagnose SAM" button (§6) matters.

---

## 5. Plan for the session

The machine being tested tomorrow is a *sister's* Windows machine —
"similar to the target" — i.e. a fresh consumer Windows box, not the
collaborator's exact machine. Treat it as a **second clean-room test**:
it may or may not reproduce. Both outcomes are informative.

**Step 0 — record the machine's starting state (do this FIRST, before
installing anything, because installing changes it):**
```bat
git --version
where git
nvidia-smi
echo %LOCALAPPDATA%
powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"
```
Capture all output. `git --version` failing here + a GPU present is the
single most valuable data point in the whole session (H2 and H1).

**Step 1 — bundle path (fast, 10 min).** Download
`EyeDataLabeller-windows-x86_64.zip` from the **v0.3.2** release,
extract, run. Does SAM enable? Check the SAM panel string and the log.

**Step 2 — source path (30–60 min).** Clone the repo, run
`deploy\install_windows.bat`, **watch the console for errors and save
the full output to a file** (this is where H2/H3 will show up):
```bat
deploy\install_windows.bat > %USERPROFILE%\Desktop\install_log.txt 2>&1
```
Note: this hides the interactive best.pt prompt — just press Enter when
it appears to seem stuck, or skip the redirect and screenshot instead.

**Step 3 — verify.** Run the checks in §4 in the created env.

**Step 4 — if it reproduces:** you have a live failing machine. Get the
traceback, match it to H1–H6, and fix the installer. If it does **not**
reproduce, the difference is on the collaborator's machine specifically
— then the deliverable is the Diagnose SAM button (§6) plus asking them
for their log file.

**Also worth doing regardless:** the app's own smoke test in the env —
```bat
python main.py --selftest
```
expects to end in `selftest: PASS` including
`selftest: SAM import chain OK (SAM_AVAILABLE)`.

---

## 6. Pending work, already designed, not yet built

**"Diagnose SAM" button — Settings → Debugging.** Offered to the user
several times, never approved, and it is the one thing that helps a
non-technical collaborator on a **bundle** install (no Python, no
command line). It should:

1. attempt `import micro_sam` and show the exact exception + traceback
   in a copyable dialog,
2. print `sys.prefix`, `torch.__version__`, `torch.cuda.is_available()`,
   the active registry entry, and whether its checkpoint exists,
3. include a "Copy to clipboard" button and the log-file path.

Ask the user before building it — but if the Windows session reproduces
nothing, this becomes the highest-value next step.

**Confirmed installer gaps to fix (independent of what tomorrow shows):**
- `install_windows.bat` has **no `git` presence check** before the
  `pip install git+https://…` step (H2). Either add a check with a clear
  error, or switch to the tarball URL and remove the git dependency
  entirely. The tarball switch is the better fix — it removes a
  requirement rather than diagnosing it.
- The CUDA swap's failure branch only warns and continues (H1). It
  should at minimum verify `import torch` still works afterwards and
  roll back to the CPU wheel if not.

Both are the same bug pattern as §7 — untested paths.

---

## 7. Project context you will need

**What this app is:** PyQt6 desktop annotator for retinal microscopy
(cells / vessels / capillaries), with SAM segmentation via micro_sam and
frame-to-frame tracking. Single user-facing repo, collaborators annotate
1000+ images.

**Current release: v0.3.2** (2026-08-06) — bundles for macOS arm64 and
Windows x86_64 attached to the GitHub release.

**Hard constraints — do not "fix" these:**
- micro-sam **must** be installed with `--no-deps`. The conda package and
  upstream's pip metadata hard-depend on **napari**, which drags in a
  second Qt stack and causes the Windows Qt/ICU DLL conflicts this
  project already fought. Its real runtime deps are pinned in
  `environment.yml` instead. If you are ever tempted to `conda install
  micro_sam` or plain `pip install micro_sam`, don't.
- `main.py` preloads ICU from `System32` on Windows on purpose.
- Various guards in the controller exist because of specific past bugs;
  read the comment before removing one.

**A very recent precedent — read this, it is the same shape as H1/H2.**
On 2026-08-06 a fully isolated install test on macOS (fake `$HOME`,
conda hidden from `PATH`) found that `install_mac.command` **always
failed on machines with no conda**: it downloaded Miniforge, then the
Miniforge installer refused to run because macOS `mktemp -t x.XXXXXX.sh`
puts the random suffix *after* `.sh`, and Miniforge requires `$0` to end
in `.sh`. Every prior test passed because every test machine already had
conda and skipped that branch. Fixed in PR #15, shipped in v0.3.2.
**Lesson: the bugs live in the branches CI and the dev machine never
take.** H1 (no GPU on CI) and H2 (git preinstalled on CI) are precisely
that.

**Working conventions the user expects:**
- Small, focused commits; PR per fix; CI must be green before merge.
- Smoke-test after changes and relaunch the app; the user reviews via
  **screenshots**, they do not run the test suite themselves.
- End each stretch of work with concrete "things to try".
- Never write test junk into the real app settings — tests that touch
  `QSettings` must use an isolated org/app name. (A scratch test once
  polluted the user's real model registry.)
- Ask before outward-facing actions (merging, tagging a release).

**Useful files:**

| Path | What |
|---|---|
| `core/sam_service.py` | the import that fails; `SAM_AVAILABLE` |
| `core/model_registry.py` | model registry, `BUILTINS`, `ensure_migrated()` |
| `core/debug.py` | `log` / `log_error`; log file creation |
| `core/app_paths.py` | log dir (`%LOCALAPPDATA%\EyeDataLabeller\logs`) |
| `deploy/install_windows.bat` | the Windows installer under suspicion |
| `main.py` | `--selftest`, ICU preload, app bootstrap |
| `.github/workflows/build.yml` | CI: micromamba env, bundle build, selftests |
| `.claude/skills/ci-bundle-debugging` | playbook for bundle/DLL failures |
| `GUIDE.md` | illustrated user guide, incl. logging/debugging §9 |

**Note on CI vs installers:** CI does **not** run `install_windows.bat`.
It builds the env with micromamba from `environment.yml`, then installs
micro-sam with the same `--no-deps` git URL. So CI validates the
*dependency set*, never the *installer script*. That gap is the whole
reason this handoff exists.

---

## 8. What to report back

1. The `micro_sam import failed` traceback block (verbatim).
2. Which hypothesis it matches, and why.
3. `git --version` / `nvidia-smi` / `pip show micro-sam` results.
4. Whether the bundle path and the source path behave differently.
5. A proposed fix as a PR — and **ask before tagging any release**.
