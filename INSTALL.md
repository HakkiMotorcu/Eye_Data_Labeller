# Installing Eye Data Labeller

Two install paths. Pick whichever fits the user.

| | Tier A — Standalone bundle | Tier B — Miniforge installer |
| --- | --- | --- |
| Terminal needed? | No, ever. | One double-click on `install_*` |
| Disk footprint | 1.5–2.5 GB | ~3.5 GB (conda env on disk) |
| Updates | Re-download new bundle | `git pull` + re-run `install_*` |
| Best for | Non-Python collaborators | Labs that update often |

The model weights (`sam_hela/best.pt`, ~400 MB) are **not bundled** —
the app downloads them on first use from a URL configured in the I/O
Settings dialog. See "Model weights" at the bottom.

---

## Tier A — One-click bundles

After the repo is pushed to GitHub and `.github/workflows/build.yml`
has run (on a tag push, e.g. `git tag v0.1.0 && git push --tags`),
three artifacts appear on the project's Releases page:

| OS | Download | What to do |
| --- | --- | --- |
| macOS (Apple Silicon) | `EyeDataLabeller-macos-arm64.zip` | Unzip → drag `EyeDataLabeller.app` into Applications → first launch: **right-click → Open** to bypass Gatekeeper warning. |
| Windows | `EyeDataLabeller-windows-x86_64.zip` | Unzip anywhere → double-click `EyeDataLabeller.exe`. SmartScreen warning: *More info* → *Run anyway*. |
| Linux  | `EyeDataLabeller-linux-x86_64.zip` | Unzip → `./EyeDataLabeller/EyeDataLabeller`. |

The first-run Gatekeeper / SmartScreen dance is unavoidable for
unsigned binaries. Code signing (Apple Developer Program $99/year or
a Windows code-signing certificate ~$300/year) removes both warnings —
worth it if the user count crosses ~10.

---

## Tier B — Miniforge installer

Best when the user has a few minutes for a one-time terminal moment
and you'll be pushing updates often.

### macOS

1. Clone or unzip the project somewhere stable (e.g.
   `~/Projects/Eye_Data_Labeller`).
2. Open Finder → navigate to `deploy/` → **right-click**
   `install_mac.command` → **Open**. Confirm "Open" when macOS warns.
3. Wait ~5–15 minutes while Miniforge installs and the env builds.
4. Double-click **EyeDataLabeller** on your Desktop.

### Windows

1. Clone or unzip the project (e.g.
   `C:\Users\<you>\Projects\Eye_Data_Labeller`).
2. Open `deploy\` → double-click `install_windows.bat`. SmartScreen:
   *More info* → *Run anyway*.
3. Double-click **EyeDataLabeller.bat** on your Desktop.

NVIDIA GPU? After install, open *Anaconda Prompt (Miniforge3)*:
```bat
conda activate eye-labeller
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(`cu118` / `cu124` etc. depending on your driver — `nvidia-smi`
shows it.)

### Linux

```bash
cd /path/to/Eye_Data_Labeller
bash deploy/install_linux.sh
```

Adds `~/Desktop/EyeDataLabeller.sh` and a CLI shim
`~/.local/bin/eye-labeller`. For NVIDIA, do the same `pip install
torch --index-url …` swap as Windows above.

---

## Model weights

`sam_hela/best.pt` (~400 MB) is downloaded on first use.

### Where the app looks

In order:

1. **In-app setting**: open **I/O panel → Output settings… → SAM-HeLa
   download URL**. Paste a public HTTPS URL.
2. **Environment variable**: `EYE_LABELLER_SAM_HELA_URL=<url>` before
   launching.
3. **Compile-time default**: `DEFAULT_SAM_HELA_URL` in
   `core/model_download.py` — set this before packaging the Tier A
   bundles so users don't need to configure anything.

### Where to host

| Host | Pros | Cons |
| --- | --- | --- |
| **Hugging Face Hub** (model repo) | Free, versioned, dead-simple HTTPS URL like `https://huggingface.co/<org>/<repo>/resolve/main/best.pt` | None really |
| **GitHub Release asset** | Free, sits next to the binaries | 2 GB per-asset cap (fine for `best.pt`) |
| **S3** with presigned URL | You own it | URL expires; bake into the bundle is fragile |
| Google Drive / Box share link | Easy to grab | `urlretrieve` chokes on their interstitial HTML — needs `gdown` or a workaround |

Recommendation: **Hugging Face**.

### Where the file lands on disk

- **Tier A bundles**: under the OS user data root —
  - macOS: `~/Library/Application Support/EyeDataLabeller/models/checkpoints/sam_hela/best.pt`
  - Windows: `%LOCALAPPDATA%\EyeDataLabeller\models\checkpoints\sam_hela\best.pt`
  - Linux: `~/.local/share/EyeDataLabeller/models/checkpoints/sam_hela/best.pt`
- **Tier B / dev runs**: `<project>/.user_data/models/checkpoints/sam_hela/best.pt`, falling back to the legacy `<project>/models/checkpoints/sam_hela/best.pt` if the file already lives there.

Downloads stream into a `.part` sibling and atomically rename on
success — Ctrl+C / crash mid-download leaves no half-file.

---

## Troubleshooting

- **"Modules import OK but the app shows a black window."** Likely a
  GPU/driver issue on Linux + Wayland. Run with
  `QT_QPA_PLATFORM=xcb python main.py`.
- **macOS: "is damaged and can't be opened."** Unsigned-bundle
  Gatekeeper false-positive. Run `xattr -dr com.apple.quarantine
  EyeDataLabeller.app` once.
- **Windows: SmartScreen still won't run it.** Right-click → Properties
  → check *Unblock* at the bottom → OK.
- **Embedding precompute too slow.** Run with --debug; if you see
  "Using cpu device", you're on the CPU PyTorch build — see the
  per-OS GPU notes above.
- **`No SAM-HeLa checkpoint URL configured`** on first SAM use. Open
  I/O Settings → paste a download URL.
