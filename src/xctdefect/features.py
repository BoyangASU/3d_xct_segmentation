"""Per-defect feature extraction from a labeled volume.

Given an integer label volume, compute geometric descriptors for every defect:
voxel volume, equivalent spherical diameter (ESD), bounding-box dimensions,
centroid, and aspect ratio. These feed the porosity analysis and the DBSCAN
clustering step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def defect_table(labels: np.ndarray, voxel_size: float = 1.0) -> pd.DataFrame:
    """Build a per-defect feature table from a labeled volume.

    Parameters
    ----------
    labels:
        Integer label volume (0 = background), shape ``(D, H, W)`` or
        ``(H, W, D)`` -- axis order only affects which centroid column is
        which.
    voxel_size:
        Physical edge length of one voxel (e.g. micrometers). Volumes are
        scaled by ``voxel_size`` and ESD is reported in millimeters.

    Returns
    -------
    pd.DataFrame
        One row per defect with columns: ``label``, ``volume``, ``esd``,
        ``bbox_width``, ``bbox_height``, ``bbox_depth``, ``centroid_0/1/2``,
        ``aspect_ratio``.
    """
    from scipy import ndimage as ndi

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids != 0]
    if label_ids.size == 0:
        return pd.DataFrame()

    objects = ndi.find_objects(labels)
    centroids = ndi.center_of_mass(np.ones_like(labels), labels, label_ids)
    counts = ndi.sum(np.ones_like(labels), labels, label_ids)

    rows = []
    for lid, centroid, count in zip(label_ids, centroids, counts):
        sl = objects[lid - 1]
        if sl is None:
            continue
        dims = [(s.stop - s.start) * voxel_size for s in sl]
        # Pad to 3 dims defensively.
        while len(dims) < 3:
            dims.append(voxel_size)
        d0, d1, d2 = dims[:3]

        volume = count * (voxel_size ** 3)
        esd = ((volume * 6 / np.pi) ** (1 / 3)) * 1e-3  # um^3 -> mm

        sorted_dims = sorted(dims[:3], reverse=True)
        aspect_ratio = sorted_dims[2] / sorted_dims[0] if sorted_dims[0] > 0 else 0.0

        rows.append(
            {
                "label": int(lid),
                "volume": volume,
                "esd": esd,
                "bbox_width": d2,
                "bbox_height": d1,
                "bbox_depth": d0,
                "centroid_0": centroid[0],
                "centroid_1": centroid[1],
                "centroid_2": centroid[2],
                "aspect_ratio": aspect_ratio,
            }
        )

    return pd.DataFrame(rows)


def global_porosity(labels: np.ndarray) -> float:
    """Fraction of voxels that belong to any defect (global porosity)."""
    total = labels.size
    defect = np.count_nonzero(labels)
    return defect / total


def equivalent_spherical_diameter(volume_um3: np.ndarray) -> np.ndarray:
    """Equivalent spherical diameter (mm) from defect volumes in um^3."""
    return ((np.asarray(volume_um3) * 6 / np.pi) ** (1 / 3)) * 1e-3
