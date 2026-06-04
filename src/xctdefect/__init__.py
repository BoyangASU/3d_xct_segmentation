"""xctdefect: 3D XCT defect segmentation and analysis for LPBF parts.

A pipeline for non-destructive internal-defect identification in laser powder
bed fusion (LPBF) parts from X-ray computed tomography (XCT):

1. :mod:`xctdefect.io`           -- load XCT TIFF slice stacks
2. :mod:`xctdefect.preprocess`   -- K-means denoising / intensity separation
3. :mod:`xctdefect.segmentation` -- 3D Voronoi-Otsu defect labeling
4. :mod:`xctdefect.features`     -- per-defect volume / ESD / bbox / porosity
5. :mod:`xctdefect.clustering`   -- DBSCAN typical-defect vs. outlier separation
6. :mod:`xctdefect.pointcloud`   -- point-cloud extraction & projections

See :func:`run_pipeline` for an end-to-end convenience wrapper.
"""

from __future__ import annotations

import numpy as np

from . import clustering, features, io, pointcloud, preprocess, segmentation

__all__ = [
    "clustering",
    "features",
    "io",
    "pointcloud",
    "preprocess",
    "segmentation",
    "run_pipeline",
]

__version__ = "1.0.0"


def run_pipeline(
    volume: np.ndarray,
    voxel_size: float = 1.0,
    spot_sigma: float = 4.0,
    outline_sigma: float = 5.0,
    eps: float = 0.04,
    min_samples: int = 6,
    denoise: bool = True,
):
    """Run the full XCT defect-analysis pipeline on a volume.

    Parameters
    ----------
    volume:
        Raw XCT volume, shape ``(H, W, D)`` (defects darker than the solid).
    voxel_size:
        Physical voxel edge length (e.g. micrometers).
    spot_sigma, outline_sigma:
        Voronoi-Otsu labeling parameters.
    eps, min_samples:
        DBSCAN parameters.
    denoise:
        Whether to apply per-slice K-means separation first.

    Returns
    -------
    dict
        ``{"labels", "table", "porosity", "cluster_labels", "cluster_summary",
        "benchmark", "typical"}``.
    """
    # 1-2. Preprocess (K-means defect/solid separation).
    vol = preprocess.kmeans_volume(volume) if denoise else volume

    # 3. 3D Voronoi-Otsu labeling.
    labels = segmentation.voronoi_otsu_labeling(vol, spot_sigma, outline_sigma)

    # 4. Per-defect features and global porosity.
    table = features.defect_table(labels, voxel_size=voxel_size)
    porosity = features.global_porosity(labels)

    # 5. DBSCAN clustering -> typical defects vs. outliers.
    cluster_labels, summary, benchmark, typical = None, None, None, None
    if not table.empty:
        feats = clustering.sorted_dimension_features(table)
        cluster_labels, summary = clustering.dbscan_cluster(feats, eps, min_samples)
        benchmark, typical = clustering.typical_defect(table, cluster_labels)

    return {
        "labels": labels,
        "table": table,
        "porosity": porosity,
        "cluster_labels": cluster_labels,
        "cluster_summary": summary,
        "benchmark": benchmark,
        "typical": typical,
    }
