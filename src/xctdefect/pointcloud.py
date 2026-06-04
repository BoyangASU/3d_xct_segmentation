"""Point-cloud extraction and projection helpers for labeled defects.

Each labeled defect region can be exported as a 3D point cloud (the voxel
coordinates belonging to that label) for visualization or downstream geometric
analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def defect_point_cloud(labels: np.ndarray, label_id: int) -> pd.DataFrame:
    """Extract the voxel coordinates of a single defect as a point cloud.

    Parameters
    ----------
    labels:
        Integer label volume.
    label_id:
        The defect label to extract.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z`` of the voxel coordinates.
    """
    coords = np.argwhere(labels == label_id)
    if coords.size == 0:
        return pd.DataFrame(columns=["x", "y", "z"])
    # argwhere returns axis order (0, 1, 2); name them z, x, y to match the
    # slice-first (D, H, W) convention used in the study.
    return pd.DataFrame({"x": coords[:, 1], "y": coords[:, 2], "z": coords[:, 0]})


def all_defects_point_cloud(labels: np.ndarray) -> pd.DataFrame:
    """Extract every defect voxel as a labeled point cloud.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, ``label``.
    """
    coords = np.argwhere(labels != 0)
    if coords.size == 0:
        return pd.DataFrame(columns=["x", "y", "z", "label"])
    return pd.DataFrame(
        {
            "x": coords[:, 1],
            "y": coords[:, 2],
            "z": coords[:, 0],
            "label": labels[coords[:, 0], coords[:, 1], coords[:, 2]],
        }
    )


def max_projections(volume: np.ndarray):
    """Maximum-intensity projections along each axis.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Projections onto the planes orthogonal to axis 0, 1, and 2.
    """
    return (
        volume.max(axis=0),
        volume.max(axis=1),
        volume.max(axis=2),
    )


def save_npy(point_cloud: pd.DataFrame, path: str):
    """Save a point cloud to a ``.npy`` file as an ``(n, 3)`` array."""
    arr = point_cloud[["x", "y", "z"]].to_numpy()
    np.save(path, arr)
