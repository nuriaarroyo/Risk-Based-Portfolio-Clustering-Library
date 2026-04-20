from __future__ import annotations

from typing import List

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


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
    # scipy expects a condensed distance vector.
    dist_condensed = squareform(dist_matrix.values, checks=False)

    # Hierarchical linkage.
    linkage_matrix = linkage(dist_condensed, method=method)

    # Cluster labels 1..n_clusters.
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    tickers = dist_matrix.index.to_list()
    clusters: list[list[str]] = []

    for cluster_id in range(1, n_clusters + 1):
        assets_in_cluster = [asset for asset, label in zip(tickers, labels) if label == cluster_id]
        if assets_in_cluster:
            clusters.append(assets_in_cluster)

    return clusters
