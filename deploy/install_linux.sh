#!/usr/bin/env bash
# Eye Data Labeller — one-shot installer for Linux (x86_64).
#
# Steps:
#   1. Install Miniforge into ~/miniforge3 if missing.
#   2. Create / update the `eye-labeller` conda env from environment.yml
#      (ships CPU PyTorch as the baseline — same on every platform).
#   3. Detect NVIDIA GPU via nvidia-smi; if present, swap PyTorch for
#      the CUDA-enabled pip wheel.
#   4. Drop a launcher script at ~/Desktop/EyeDataLabeller.sh (if a
#      Desktop folder exists) and a CLI shim at ~/.local/bin/eye-labeller.
#
# CUDA WHEEL VERSION:
#   We default to cu124 (CUDA 12.4 runtime) — matches PyTorch's
#   recommended default and works with any NVIDIA driver >=550.
#   If your driver is older (RHEL/CentOS labs sometimes have 525),
#   edit CUDA_INDEX below to cu121 instead. Check your driver with
#   `nvidia-smi` and look at the "Driver Version:" line. Mapping:
#     driver >=550 → cu124
#     driver >=525 → cu121
#     driver <525  → talk to IT, your machine is overdue for updates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="eye-labeller"
DESKTOP_LAUNCHER="$HOME/Desktop/EyeDataLabeller.sh"
CLI_SHIM="$HOME/.local/bin/eye-labeller"

# CUDA wheel index — change to cu121 for older drivers (see header).
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

say()  { printf "\033[1;36m[install]\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m[install] FAILED:\033[0m %s\n" "$*" >&2; exit 1; }

# ---- 1. Miniforge ---------------------------------------------------
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
    say "Installing Miniforge into $MINIFORGE_DIR"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    tmpfile="$(mktemp -t miniforge.XXXXXX.sh)"
    curl -fL "$url" -o "$tmpfile" || fail "Download of Miniforge failed"
    bash "$tmpfile" -b -p "$MINIFORGE_DIR" || fail "Miniforge install failed"
    rm -f "$tmpfile"
else
    say "Miniforge already at $MINIFORGE_DIR — reusing it."
fi

# shellcheck source=/dev/null
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# ---- 2. Conda env ---------------------------------------------------
ENV_YML="$PROJECT_ROOT/environment.yml"
[ -f "$ENV_YML" ] || fail "environment.yml not found at $ENV_YML"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "Updating existing '$ENV_NAME' env from $ENV_YML"
    conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
    say "Creating '$ENV_NAME' env from $ENV_YML  (5-15 min)"
    conda env create -n "$ENV_NAME" -f "$ENV_YML"
fi

# ---- 3. GPU detection + CUDA PyTorch swap ---------------------------
# env.yml ships CPU PyTorch as the baseline so every platform has the
# same starting point. On Linux + NVIDIA we swap to the CUDA wheel.
# --force-reinstall to overwrite conda's CPU torch; --no-deps so we
# don't disturb numpy / scipy / BLAS pin / etc.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    say "NVIDIA GPU detected: $gpu_name  (driver $driver_ver)"
    say "Swapping to CUDA-enabled PyTorch from $CUDA_INDEX"
    conda activate "$ENV_NAME"
    pip install --force-reinstall --no-deps \
        torch torchvision --index-url "$CUDA_INDEX" || \
        say "(CUDA PyTorch install FAILED — app will fall back to CPU. Check driver/CUDA compatibility per the header notes in this script.)"
    conda deactivate
else
    say "No NVIDIA GPU detected — keeping CPU PyTorch."
    say "  (App will use CPU for SAM inference. Slower but works.)"
fi

# ---- 4. Launchers ---------------------------------------------------
if [ -d "$HOME/Desktop" ]; then
    say "Writing desktop launcher to $DESKTOP_LAUNCHER"
    cat >"$DESKTOP_LAUNCHER" <<EOF
#!/usr/bin/env bash
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$PROJECT_ROOT"
exec python main.py "\$@"
EOF
    chmod +x "$DESKTOP_LAUNCHER"
fi

mkdir -p "$(dirname "$CLI_SHIM")"
say "Writing CLI shim to $CLI_SHIM"
cat >"$CLI_SHIM" <<EOF
#!/usr/bin/env bash
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$PROJECT_ROOT"
exec python main.py "\$@"
EOF
chmod +x "$CLI_SHIM"

# ---- 5. Optional SAM-HeLa checkpoint path ---------------------------
echo ""
say "Optional: if you already have sam_hela/best.pt on disk, point the"
say "app at it now. Press Enter to skip — you can set it later via the"
say "app's I/O > Output settings… dialog."
printf "  Path to best.pt (or empty to skip): "
read MODEL_PATH || MODEL_PATH=""
if [ -n "$MODEL_PATH" ]; then
    conda activate "$ENV_NAME"
    python "$SCRIPT_DIR/configure_model.py" "$MODEL_PATH" || \
        say "(path not saved; you can set it later in the app settings)"
fi

say "Done. Launch via the Desktop icon or by running 'eye-labeller'."
