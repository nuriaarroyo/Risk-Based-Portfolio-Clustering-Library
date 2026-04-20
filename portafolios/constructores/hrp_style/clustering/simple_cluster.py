from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(slots=True)
class HierarchicalClusteringResult:
    clusters: list[list[str]]
    linkage_matrix: np.ndarray
    labels: pd.Series
    method: str


def hierarchical_cluster_details(
    dist_matrix: pd.DataFrame,
    n_clusters: int = 3,
    method: str = "average",
) -> HierarchicalClusteringResult:
    """
    Run hierarchical clustering and keep the full clustering diagnostics.
    """
    dist_condensed = squareform(dist_matrix.values, checks=False)
    linkage_matrix = linkage(dist_condensed, method=method)
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    tickers = dist_matrix.index.to_list()
    clusters: list[list[str]] = []
    for cluster_id in range(1, n_clusters + 1):
        assets_in_cluster = [asset for asset, label in zip(tickers, labels) if label == cluster_id]
        if assets_in_cluster:
            clusters.append(assets_in_cluster)

    return HierarchicalClusteringResult(
        clusters=clusters,
        linkage_matrix=np.asarray(linkage_matrix, dtype=float),
        labels=pd.Series(labels, index=tickers, name="cluster_label"),
        method=method,
    )


def hierarchical_clusters(
    dist_matrix: pd.DataFrame,
    n_clusters: int = 3,
    method: str = "average",
) -> List[list[str]]:
    """
    Run hierarchical clustering on the distance matrix and return
    a list of clusters (each cluster is a list of tickers).

    dist_matrix: square DataFrame (assets x assets)
    n_clusters: desired number of clusters
    """
    return hierarchical_cluster_details(
        dist_matrix,
        n_clusters=n_clusters,
        method=method,
    ).clusters
