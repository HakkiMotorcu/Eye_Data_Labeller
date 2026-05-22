# Eye Data Labeller — Deployment

Two ways to install on a collaborator's machine.

## Tier A — One-click standalone bundle (recommended for non-developers)

> Coming soon — built per OS by the GitHub Actions workflow at
> `.github/workflows/build.yml`. Once that workflow has run, the
> Releases page will host:
>
> - **macOS**: `EyeDataLabeller-<version>-mac.dmg` (Apple Silicon)
> - **Windows**: `EyeDataLabeller-<version>-windows.exe`
> - **Linux**: `EyeDataLabeller-<version>-linux.AppImage`
>
> Users download, double-click, run. No terminal, no conda.
> See `INSTALL.md` (project root) for first-run details (unsigned
> binary warnings, model URL).

## Tier B — Miniforge + double-click launcher (recommended for developers / labs)

Smaller download, easier to update later (`git pull` + `install_*` again),
but requires one terminal moment during install.

### macOS

1. Make sure the project folder is on disk somewhere (e.g.
   `~/Projects/Eye_Data_Labeller`). A `git clone` works fine.
2. Open Finder, navigate to `deploy/`, and **double-click**
   `install_mac.command`.
   - First time only: macOS may say *"can't be opened because Apple
     cannot check it for malicious software."* Right-click the file
     instead and choose **Open** → **Open** again to confirm.
3. The script installs Miniforge into `~/miniforge3`, creates the
   `eye-labeller` conda environment, and writes
   `~/Desktop/EyeDataLabeller.command`.
4. Double-click that desktop launcher to run the app.

### Linux

```bash
cd /path/to/Eye_Data_Labeller
bash deploy/install_linux.sh
```

Adds `~/Desktop/EyeDataLabeller.sh` and a CLI shim
`~/.local/bin/eye-labeller`. NVIDIA-GPU users should follow the
post-install note in the script to swap in CUDA-PyTorch wheels.

### Windows

1. Copy / clone the project folder somewhere stable (e.g.
   `C:\Users\<you>\Projects\Eye_Data_Labeller`).
2. Open `deploy\` in File Explorer and **double-click**
   `install_windows.bat`.
   - Windows Defender SmartScreen may warn. Click *More info* →
     *Run anyway*.
3. The script installs Miniforge into `%USERPROFILE%\miniforge3`,
   creates the `eye-labeller` conda environment, and writes
   `%USERPROFILE%\Desktop\EyeDataLabeller.bat`.
4. Double-click that desktop launcher to run the app.

NVIDIA-GPU users: post-install, open *Anaconda Prompt (Miniforge3)*
and run:
```bat
conda activate eye-labeller
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(use `cu118` / `cu124` as appropriate for your driver).

## Model weights

The fine-tuned SAM-HeLa checkpoint (`models/checkpoints/sam_hela/best.pt`,
~400 MB) is **not bundled** with either tier — it's downloaded on first
use from a URL the deployer configures.

Set the URL once per workstation:

- **In the app**: open **I/O → Output settings…** and paste a public
  HTTPS URL (Hugging Face download URL, GitHub Release asset URL, S3
  presigned URL) into "SAM-HeLa download URL". The app stores it via
  QSettings.
- **Via environment**: set `EYE_LABELLER_SAM_HELA_URL=<url>` before
  launching.
- **At build time**: edit `DEFAULT_SAM_HELA_URL` in
  `core/model_download.py` and re-build.

Recommended hosting:

- **Hugging Face**: create a public *model* repo, upload `best.pt`,
  use the "resolve/main/best.pt" raw URL.
- **GitHub Releases**: attach `best.pt` to a Release on this repo and
  copy the asset URL.
- **University Box / S3 / Dropbox**: needs a direct download URL —
  Box and Drive share links won't work without API tweaks.
