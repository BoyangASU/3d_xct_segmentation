"""3D defect-region segmentation via Voronoi-Otsu labeling.

The denoised volume is segmented into individual labeled defect regions by
combining three classic steps:

1. **Gaussian blur + spot detection** -- find one seed per defect (local
   maxima after blurring with ``spot_sigma``).
2. **Otsu thresholding** -- a binary defect/solid mask (blurred with
   ``outline_sigma`` first).
3. **Voronoi labeling** -- grow each seed into a region, then intersect with
   the binary mask so every connected defect gets a unique integer label.

A GPU implementation (pyclesperanto's ``voronoi_otsu_labeling``) is used when
available; otherwise a pure scikit-image / SciPy fallback is provided so the
package runs anywhere.
"""

from __future__ import annotations

import numpy as np


def voronoi_otsu_labeling(volume: np.ndarray, spot_sigma: float = 4.0, outline_sigma: float = 5.0):
    """Label individual 3D defect regions in a volume.

    Tries the GPU backend (pyclesperanto) first and falls back to a CPU
    implementation built on scikit-image and SciPy.

    Parameters
    ----------
    volume:
        3D grayscale volume (defect = bright after preprocessing). Any axis
        order works as long as it is consistent downstream.
    spot_sigma:
        Blur scale controlling seed (spot) detection -- larger merges nearby
        defects into one seed.
    outline_sigma:
        Blur scale for the Otsu mask outline.

    Returns
    -------
    np.ndarray
        Integer label volume, same shape as ``volume`` (0 = background).
    """
    try:
        import pyclesperanto_prototype as cle

        labels = cle.voronoi_otsu_labeling(
            volume, spot_sigma=spot_sigma, outline_sigma=outline_sigma
        )
        return np.asarray(cle.pull(labels)).astype(np.int32)
    except Exception:
        return _voronoi_otsu_cpu(volume, spot_sigma, outline_sigma)


def _voronoi_otsu_cpu(volume, spot_sigma, outline_sigma):
    """CPU fallback for Voronoi-Otsu labeling."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    vol = volume.astype(float)

    # 1. Spot detection: maxima of the blurred image.
    blurred_spots = ndi.gaussian_filter(vol, spot_sigma)
    coords = peak_local_max(blurred_spots, min_distance=1)
    seeds = np.zeros(vol.shape, dtype=int)
    for i, c in enumerate(coords, start=1):
        seeds[tuple(c)] = i

    # 2. Otsu binary mask of the outline-blurred image.
    blurred_outline = ndi.gaussian_filter(vol, outline_sigma)
    nz = blurred_outline[blurred_outline > 0]
    thresh = threshold_otsu(nz) if nz.size else 0
    mask = blurred_outline > thresh

    # 3. Watershed (a Voronoi-like partition) constrained to the mask.
    labels = watershed(-blurred_spots, markers=seeds, mask=mask)
    return labels.astype(np.int32)
