# portafolios/constructores/hrp_style/clustering/simple_hierarchical.py
from __future__ import annotations

from typing import List
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def hierarchical_clusters(
    dist_matrix: pd.DataFrame,
    n_clusters: int = 3,
    method: str = "average",
) -> List[list[str]]:
    """
    Hace clustering jerárquico sobre la matriz de distancias y regresa
    una lista de clusters (cada cluster es una lista de tickers).

    dist_matrix: DataFrame cuadrado (assets x assets)
    n_clusters: número de clusters deseados
    """
    # scipy espera un vector "condensed" de distancias
    dist_condensed = squareform(dist_matrix.values, checks=False)

    # linkage jerárquico
    Z = linkage(dist_condensed, method=method)

    # etiquetas de cluster 1..n_clusters
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    tickers = dist_matrix.index.to_list()
    clusters: list[list[str]] = []

    for k in range(1, n_clusters + 1):
        assets_k = [asset for asset, lab in zip(tickers, labels) if lab == k]
        if assets_k:
            clusters.append(assets_k)

    return clusters
