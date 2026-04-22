from __future__ import annotations

from typing import Optional

import pandas as pd

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def get_distance_matrix(
    hrp: HRPStyle,
    reorder_by_clusters: bool = True,
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the HRP distance matrix from the latest run.

    If `reorder_by_clusters` is True, rows and columns are reordered by cluster.
    `file_path` is accepted for API compatibility but ignored because this helper
    returns the raw matrix instead of a figure.
    """

    _ = file_path

    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("This HRPStyle instance does not expose 'last_dist'. Run the HRP construction step first.")

    distance_matrix: pd.DataFrame = hrp.last_dist.copy()

    if reorder_by_clusters and hasattr(hrp, "last_clusters"):
        ordered_assets = [asset for cluster in hrp.last_clusters for asset in cluster]
        ordered_assets = [asset for asset in ordered_assets if asset in distance_matrix.index]
        distance_matrix = distance_matrix.loc[ordered_assets, ordered_assets]

    return distance_matrix


def get_distmat(
    hrp: HRPStyle,
    ordenar_por_clusters: bool = True,
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the original helper name.
    """

    return get_distance_matrix(hrp, reorder_by_clusters=ordenar_por_clusters, file_path=file_path)


__all__ = ["get_distance_matrix", "get_distmat"]
