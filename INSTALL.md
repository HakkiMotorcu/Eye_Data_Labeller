# Installing Eye Data Labeller

## TL;DR for collaborators

```bash
git clone https://github.com/HakkiMotorcu/Eye_Data_Labeller.git
cd Eye_Data_Labeller
# macOS:
deploy/install_mac.command          # right-click → Open if Gatekeeper warns
# Linux:
bash deploy/install_linux.sh
# Windows (in cmd.exe):
deploy\install_windows.bat
```

The installer drops a launcher on your Desktop. Double-click it. The
app opens on a landing page; open a stack (TIFF or video) from there.

**Add the SAM model (one-time):** the ~400 MB `best.pt` checkpoint
does **not** ship with the app. Get it from whoever runs your lab (a
shared drive / Box / USB), then in the app: **Model → Add model…**,
give it a tag, and browse to the `best.pt`. That's it — it's
remembered across launches. Without a model, manual annotation still
works; only the one-click **SAM Box** is disabled.

If you have an NVIDIA GPU on Linux/Windows, the installer auto-detects
it and swaps in CUDA-enabled PyTorch. On macOS, Apple Silicon GPU (MPS)
support is automatic.

**Prefer no terminal?** Download a pre-built bundle from the
[Releases page](https://github.com/HakkiMotorcu/Eye_Data_Labeller/releases)
instead — see [Standalone bundles](#standalone-bundles-alternative-install)
below.

---

## What you need before you start

- **A machine you can install software on.** No admin password needed —
  everything goes under your home directory.
- **~3.5 GB free disk** for the conda environment.
- **Internet** for the initial install (downloads conda + dependencies).
- **A `best.pt` SAM-HeLa checkpoint file** *(optional)* — register it
  later via **Model → Add model…** in the app. Without it, manual
  annotation works; only SAM Box is disabled.

---

## Per-OS install

### macOS (Apple Silicon)

1. **Clone the repo** somewhere stable, e.g. `~/Projects/Eye_Data_Labeller`.
2. **Finder → navigate to `deploy/` → right-click `install_mac.command`
   → Open.** Confirm "Open" when macOS warns about an unsigned script.
3. Wait ~5–15 minutes for Miniforge install + conda env create.
4. (Optional) When prompted, paste the path to your `best.pt` so SAM
   works out of the box. Hit Enter to skip.
5. Double-click **EyeDataLabeller** on your Desktop.

**Apple Silicon GPU (MPS):** automatic. Nothing to configure.

### Linux (x86_64)

```bash
git clone https://github.com/HakkiMotorcu/Eye_Data_Labeller.git
cd Eye_Data_Labeller
bash deploy/install_linux.sh
```

Adds `~/Desktop/EyeDataLabeller.sh` (if you have a Desktop folder) and
a CLI shim `~/.local/bin/eye-labeller` you can run from anywhere.

**NVIDIA GPU:** detected automatically. The installer prints
`NVIDIA GPU detected: <model> (driver <X>)` and swaps to CUDA-enabled
PyTorch. If your driver is older than 550 (`nvidia-smi` shows it),
edit `CUDA_INDEX` in the script — see the header comment for the
mapping.

### Windows (x86_64)

1. Clone the repo (e.g. via GitHub Desktop or `git clone` in cmd).
2. Open File Explorer → `deploy\` → double-click `install_windows.bat`.
   If SmartScreen warns: *More info* → *Run anyway*.
3. Wait for Miniforge + env install.
4. (Optional) Paste your `best.pt` path when prompted.
5. Double-click **EyeDataLabeller** on your Desktop.

**NVIDIA GPU:** detected via `nvidia-smi` from the cmd shell. Same
auto-swap behavior as Linux.

---

## SAM-HeLa model weights (`best.pt`)

The ~400 MB checkpoint isn't bundled with the app (git doesn't handle
large binaries well). The app needs to find it before SAM works.

### Easiest: drop the file in the magic folder

After cloning the repo, drop your `best.pt` here:

```
models/checkpoints/sam_hela/best.pt
```

(The folder already exists with a `README.md` placeholder telling you
the same thing.) The app checks this exact path on startup. No
configuration needed.

If you don't have a `best.pt` yet, ask whoever runs your lab — they
should have it on a USB / Box / Drive / network share, or hand you a
`models.zip` containing it.

### Recommended: register it in the app (the model registry)

Works for both install paths (Tier A bundle and Tier B source):

- **Model → Add model…** → give it a tag (e.g. `sam_hela`), pick the
  base architecture (`vit_b` for the standard SAM-HeLa fine-tune), and
  browse to `best.pt`. It's remembered across launches, and you can
  register several and switch between them with the sidebar combo.
- **Settings → SAM Model** shows the full registry — add / edit /
  remove models and choose which is active.

The file is read in place — nothing is copied, so the original can
live anywhere readable (network drive, read-only volume, etc.). Tags
and file paths are each unique, so two models can't collide.

### Other ways to point at the file

- **At install time:** every installer prompts —
  `Path to best.pt (or empty to skip):`. Paste the full path (it's
  registered for you).
- **Via env var:** `EYE_LABELLER_SAM_HELA_LOCAL_PATH=/full/path` before
  launching (useful on shared lab machines) — migrated into the
  registry on first run.

### If your lab has a public download URL for `best.pt`

- **In the app:** Settings → SAM Model → **Download URL** → paste a
  public HTTPS URL (used by the built-in variants on first use).
- **Via env var:** `EYE_LABELLER_SAM_HELA_URL=<url>` before launching.

The app downloads on first SAM use, streams into a `.part` sibling
file, and atomically renames on success (so Ctrl+C never leaves you
with a half-file). Downloads land under your per-user data dir:

- macOS: `~/Library/Application Support/EyeDataLabeller/models/checkpoints/sam_hela/best.pt`
- Linux: `~/.local/share/EyeDataLabeller/models/checkpoints/sam_hela/best.pt`
- Windows: `%LOCALAPPDATA%\EyeDataLabeller\models\checkpoints\sam_hela\best.pt`

### Resolution order at runtime

1. The **active model** in the registry (Model → Add model / Settings
   → SAM Model), including a path migrated from the old setting / env
   var
2. Configured download URL → downloads on first use (built-in variants)
3. SAM stays off with a status-line hint — manual annotation still
   works; register a model to enable SAM Box

---

## Updating

The installer scripts are idempotent — re-run them after a `git pull`
to update your env:

```bash
cd Eye_Data_Labeller
git pull
deploy/install_mac.command       # or install_linux.sh / install_windows.bat
```

It detects the existing env and does `conda env update --prune` instead
of recreating. Takes a couple of minutes if no major deps changed.

---

## Troubleshooting

### macOS

- **`install_mac.command` won't open ("damaged or can't be opened"):**
  `xattr -dr com.apple.quarantine install_mac.command` then try again.
- **App launches but I see nothing:** the window can open behind
  other apps — check Mission Control / cmd-Tab for `Python` or
  `python3.12`. (It opens on a landing page, not a file dialog.)
- **SAM is using CPU and is slow:** in a terminal, `conda activate
  eye-labeller && python -c "import torch; print(torch.backends.mps.is_available())"`.
  Should print `True`. If `False`, your PyTorch wasn't built with MPS
  — shouldn't happen on Apple Silicon with conda-forge — re-run the
  installer to refresh.

### Linux

- **App launches but I see nothing:** the window (landing page)
  should open on your active monitor; check other workspaces.
- **Wayland: black window:** `QT_QPA_PLATFORM=xcb eye-labeller` (forces
  X11 instead of Wayland — Qt6 Wayland support is still patchy).
- **CUDA install failed:** the installer prints a clear "FAILED" line
  and falls back to CPU. Run `nvidia-smi` to check your driver. If
  driver < 550, edit `CUDA_INDEX` in `deploy/install_linux.sh` to
  `cu121` and re-run.

### Windows

- **SmartScreen won't run the .bat:** right-click the .bat →
  Properties → check *Unblock* at the bottom → OK.
- **CUDA install failed:** same as Linux — check `nvidia-smi`, swap
  `CUDA_INDEX` if driver is older than 550.
- **"conda not recognized" after install:** the installer scopes
  Miniforge to itself and doesn't add to PATH globally. Use the
  Desktop launcher (which activates the env) rather than typing
  `conda` in a fresh terminal.

### Anywhere

- **`No SAM-HeLa checkpoint`:** **Model → Add model…** and browse to
  your `best.pt` (or Settings → SAM Model → Download URL for a
  built-in variant).
- **Embedding precompute too slow:** check your device with
  `conda activate eye-labeller && python -c "from core.device import
  describe_device; print(describe_device())"`. Should print `cuda
  (...)` or `mps (...)`. If `cpu`, the install-time GPU swap didn't
  take effect.

---

## Standalone bundles (alternative install)

For collaborators who don't want to touch a terminal at all, we
also ship `.app` / `.exe` / Linux standalone bundles via PyInstaller.
Built automatically by `.github/workflows/build.yml` on every tag
push (`git tag vX.Y.Z && git push --tags`) and attached to the
matching GitHub Release.

**Status (as of v0.3.0):**

| Platform | Status |
| --- | --- |
| macOS Apple Silicon | ✓ CI builds the bundle and runs a headless launch (`--selftest`) on it every release |
| Windows x86_64 | ✓ Same — CI builds + headless-launches the bundle |
| Linux x86_64 | Source install only (no bundle job in CI) — use `deploy/install_linux.sh` |

### Downloading a bundle

1. Go to the [Releases page](https://github.com/HakkiMotorcu/Eye_Data_Labeller/releases).
2. Download the zip for your platform.
3. Unzip → drag `EyeDataLabeller.app` into Applications (mac) /
   unzip anywhere (Win/Linux).
4. First launch: macOS Gatekeeper will warn — **right-click → Open**
   to bypass. On Windows: SmartScreen → *More info* → *Run anyway*.

Bundle limitations vs. Tier B install:
- **GPU support is whatever PyTorch's CPU wheel includes.** No CUDA
  in bundles (Win/Linux). On macOS, MPS works (it's runtime-detected,
  no special build needed).
- **Model weights (`best.pt`, ~400 MB) are NOT in the bundle.** After
  launch, register it via **Model → Add model…** (same as the Tier B
  install). Until then, manual annotation works and SAM Box is off.
- **Larger disk footprint** (~600 MB unzipped vs. Tier B's ~3.5 GB
  shared conda env — but the bundle is fully self-contained).
