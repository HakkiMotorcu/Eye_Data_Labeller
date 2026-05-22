#!/usr/bin/env bash
# Eye Data Labeller — one-shot installer for Linux (x86_64).
#
# Steps:
#   1. Install Miniforge into ~/miniforge3 if missing.
#   2. Create / update the `eye-labeller` conda env from environment.yml.
#   3. Drop a launcher script at ~/Desktop/EyeDataLabeller.sh (if a
#      Desktop folder exists) and a CLI shim at ~/.local/bin/eye-labeller.
#
# For NVIDIA users: after install, run
#     conda activate eye-labeller
#     pip uninstall torch torchvision
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# to swap in the CUDA-enabled PyTorch wheels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="eye-labeller"
DESKTOP_LAUNCHER="$HOME/Desktop/EyeDataLabeller.sh"
CLI_SHIM="$HOME/.local/bin/eye-labeller"

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

# ---- 3. Launchers ---------------------------------------------------
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

# ---- 4. Optional SAM-HeLa checkpoint path ---------------------------
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
