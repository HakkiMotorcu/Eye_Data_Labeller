"""One-stop device picker for any code that loads a model.

Returns the best torch device available, in priority order:

    1. CUDA  — NVIDIA GPU + CUDA-enabled PyTorch wheel installed
    2. MPS   — Apple Silicon Mac (M1/M2/M3/M4) running PyTorch >=1.12
    3. CPU   — fallback that always works

Usage:

    from core.device import pick_best_device
    device = pick_best_device()
    model.to(device)

Override via env var ``EYE_LABELLER_DEVICE=cpu`` (or ``cuda`` / ``mps``)
when you need to force one — useful for debugging GPU-specific bugs by
reproducing them on CPU, or for benchmarking.
"""

from __future__ import annotations

import os


_ENV_VAR = "EYE_LABELLER_DEVICE"


def pick_best_device() -> str:
    """Return a device string suitable for ``torch.device(...)`` / ``.to()``.

    Caller is responsible for actually applying it. We don't import
    ``torch`` at module-top so this file stays cheap to import from
    spots that don't need a model (e.g. project-loading code).
    """
    override = os.environ.get(_ENV_VAR, "").strip().lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    if override:
        # Bad value — log but don't crash; fall through to auto-detect.
        print(f"[device] ignoring invalid {_ENV_VAR}={override!r}")

    try:
        import torch  # local import: this module shouldn't drag torch in
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # MPS check guarded by hasattr — older torch builds (pre-1.12) don't
    # have the attribute at all, and conda-forge on x86_64 builds skip
    # MPS entirely.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str | None = None) -> str:
    """Human-readable description of a device, for log/UI display.

    Examples:
        'cuda (NVIDIA RTX 3090, 24 GB)'
        'mps (Apple Silicon GPU)'
        'cpu'
    """
    if device is None:
        device = pick_best_device()

    if device == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return device

    if device == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"cuda ({name}, {mem_gb:.0f} GB)"
        except Exception:
            return "cuda"
    if device == "mps":
        return "mps (Apple Silicon GPU)"
    return device
