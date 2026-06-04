"""Density-based clustering of defects (DBSCAN) and typical-defect extraction.

Defects are clustered in a 3D feature space of sorted bounding-box dimensions
``[max, mid, min]``. DBSCAN separates the dense core of *typical* defects from
sparse *outliers* (label ``-1``). The mean dimensions of the typical defects
define a benchmark defect for each sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sorted_dimension_features(table: pd.DataFrame) -> pd.DataFrame:
    """Build the ``[max, mid, min]`` sorted-dimension feature table.

    For each defect the three bounding-box edge lengths are sorted descending,
    so the features are rotation-invariant descriptors of elongation/size.
    """
    dims = table[["bbox_width", "bbox_height", "bbox_depth"]].to_numpy()
    sorted_dims = np.sort(dims, axis=1)[:, ::-1]  # descending
    out = pd.DataFrame(sorted_dims, columns=["max", "mid", "min"], index=table.index)
    return out


def dbscan_cluster(features: pd.DataFrame, eps: float = 0.04, min_samples: int = 6):
    """Cluster defects with DBSCAN on min-max-scaled features.

    Parameters
    ----------
    features:
        Table with columns ``["max", "mid", "min"]``.
    eps:
        Neighborhood radius (choose from a k-distance elbow plot). The original
        study used 0.04 on min-max-scaled features.
    min_samples:
        Minimum points to form a dense region. A common heuristic is
        ``2 * n_features`` (here 6).

    Returns
    -------
    (np.ndarray, dict)
        Per-defect cluster labels (``-1`` marks outliers) and a summary dict
        with cluster counts and the number of clusters found.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import MinMaxScaler

    X = MinMaxScaler().fit_transform(features[["max", "mid", "min"]].to_numpy())
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)

    from collections import Counter

    counts = Counter(labels)
    summary = {
        "counts": dict(counts),
        "n_clusters": len(counts) - (1 if -1 in counts else 0),
        "n_outliers": int(counts.get(-1, 0)),
    }
    return labels, summary


def kdistance(features: pd.DataFrame, k: int = 6) -> np.ndarray:
    """Sorted k-th nearest-neighbor distances for choosing ``eps``.

    Plot the returned array and look for the elbow to pick ``eps``.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler

    X = MinMaxScaler().fit_transform(features[["max", "mid", "min"]].to_numpy())
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])


def typical_defect(table: pd.DataFrame, cluster_labels: np.ndarray):
    """Mean bounding-box dimensions of the non-outlier (typical) defects.

    Returns
    -------
    (dict, pd.DataFrame)
        A ``{"width", "height", "depth"}`` benchmark and the filtered table of
        typical defects.
    """
    typical = table[cluster_labels != -1]
    benchmark = {
        "width": float(typical["bbox_width"].mean()),
        "height": float(typical["bbox_height"].mean()),
        "depth": float(typical["bbox_depth"].mean()),
    }
    return benchmark, typical


def silhouette(features: pd.DataFrame, cluster_labels: np.ndarray) -> float:
    """Silhouette score over the clustered (non-outlier) points."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler

    mask = cluster_labels != -1
    if mask.sum() < 2 or len(set(cluster_labels[mask])) < 2:
        return float("nan")
    X = MinMaxScaler().fit_transform(features[["max", "mid", "min"]].to_numpy())
    return float(silhouette_score(X[mask], cluster_labels[mask]))
