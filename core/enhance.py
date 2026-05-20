"""Display-only image enhancements: CLAHE, Frangi vesselness, gamma, invert.

These never mutate the source frames — they take a uint8 ``(H, W)`` image
and return a uint8 ``(H, W)`` image. The MainWindow display pipeline composes
them on top of either the live frame or a temporal projection.
"""

import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    """Contrast-Limited Adaptive Histogram Equalization on a uint8 image.

    Uses skimage's equalize_adapthist; result is rescaled to uint8.
    """
    from skimage import exposure
    img = image.astype(np.float32) / 255.0
    out = exposure.equalize_adapthist(
        img,
        kernel_size=max(1, int(image.shape[0] // tile_grid_size)),
        clip_limit=clip_limit / 100.0,  # skimage uses 0..1 range; map slider
    )
    return (out * 255.0).astype(np.uint8)


def frangi_vesselness(image, sigma_min=1.0, sigma_max=4.0, n_sigmas=4, black_ridges=False):
    """Multi-scale Frangi vesselness, returning a uint8 vesselness map.

    Vessels in AOSLO phase-contrast are bright on a dark background, so
    ``black_ridges=False`` is the right default. Output is rescaled to
    uint8 for direct display.
    """
    from skimage import filters
    img = image.astype(np.float32) / 255.0
    sigmas = np.linspace(sigma_min, sigma_max, max(1, int(n_sigmas)))
    out = filters.frangi(img, sigmas=sigmas, black_ridges=black_ridges)
    mn, mx = float(out.min()), float(out.max())
    if mx <= mn:
        return np.zeros_like(image, dtype=np.uint8)
    return ((out - mn) / (mx - mn) * 255.0).astype(np.uint8)


def apply_gamma(image, gamma):
    """Standard gamma correction. gamma > 1 brightens midtones; < 1 darkens."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    img = image.astype(np.float32) / 255.0
    out = np.power(img, 1.0 / max(gamma, 1e-6))
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def invert(image):
    return 255 - image
