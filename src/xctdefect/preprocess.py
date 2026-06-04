"""Preprocessing: K-means intensity separation for XCT slices.

Before 3D labeling, each slice is denoised by separating defect (dark) pixels
from the solid (bright) material with a 1D 2-means on pixel intensities. This
suppresses speckle while preserving defect regions, and is the segmentation
step labeled ``*a`` in the method pipeline.
"""

from __future__ import annotations

import numpy as np


def kmeans_separation(image: np.ndarray, max_iter: int = 100, tol: float = 1e-3):
    """Separate an image into defect (low) and solid (high) intensity classes.

    Implements Lloyd's 2-means on the non-zero pixel intensities. Returns the
    *defect* intensity map (low-intensity cluster kept, others zeroed), which
    is what the downstream labeling consumes.

    Parameters
    ----------
    image:
        2D grayscale slice.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``defect_map`` (original intensities where the pixel is in the low
        cluster, 0 elsewhere) and the two cluster centers ``(low, high)``.
    """
    nonzero = image[image.nonzero()]
    if nonzero.size == 0:
        return np.zeros_like(image), np.array([0.0, 0.0])

    low = float(nonzero.min())
    high = float(nonzero.max())
    flat = image.astype(float)

    for _ in range(max_iter):
        d_low = np.abs(flat - low)
        d_high = np.abs(flat - high)
        in_low = d_low <= d_high

        new_low = flat[in_low & (flat > 0)].mean() if (in_low & (flat > 0)).any() else low
        new_high = flat[~in_low].mean() if (~in_low).any() else high

        shift = abs(new_low - low) + abs(new_high - high)
        low, high = new_low, new_high
        if shift < tol:
            break

    defect_map = np.where((np.abs(flat - low) <= np.abs(flat - high)) & (flat > 0), image, 0)
    return defect_map.astype(image.dtype), np.array([low, high])


def kmeans_volume(volume: np.ndarray):
    """Apply :func:`kmeans_separation` slice-by-slice over a ``(H, W, D)`` volume."""
    out = np.zeros_like(volume)
    for d in range(volume.shape[2]):
        out[:, :, d], _ = kmeans_separation(volume[:, :, d])
    return out


def median_denoise(volume: np.ndarray, ksize: int = 1):
    """Light median filter applied per slice (speckle suppression)."""
    import cv2

    out = np.zeros_like(volume)
    for d in range(volume.shape[2]):
        out[:, :, d] = cv2.medianBlur(volume[:, :, d], ksize)
    return out
