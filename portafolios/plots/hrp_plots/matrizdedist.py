from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def plot_distance_matrix(
    hrp: HRPStyle,
    reorder_by_clusters: bool = True,
    file_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot the HRP distance matrix as a heatmap.

    Parameters
    ----------
    hrp
        `HRPStyle` object that has already been used in the HRP construction step.
    reorder_by_clusters
        If True, reorder rows and columns by the latest cluster ordering when available.
    file_path
        Optional HTML path where the figure should be saved.
    """

    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("This HRPStyle instance does not expose 'last_dist'. Run the HRP construction step first.")

    distance_matrix: pd.DataFrame = hrp.last_dist.copy()

    if reorder_by_clusters and hasattr(hrp, "last_clusters"):
        ordered_assets = [asset for cluster in hrp.last_clusters for asset in cluster]
        ordered_assets = [asset for asset in ordered_assets if asset in distance_matrix.index]
        distance_matrix = distance_matrix.loc[ordered_assets, ordered_assets]

    fig = go.Figure(
        data=go.Heatmap(
            z=distance_matrix.values,
            x=list(distance_matrix.columns),
            y=list(distance_matrix.index),
            colorscale="RdBu",
            colorbar=dict(title="Distance"),
        )
    )

    fig.update_layout(
        title="De Prado Distance Matrix",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        width=800,
        height=700,
    )

    if file_path is not None:
        fig.write_html(file_path)

    fig.show()
    return fig


def matriz_distancias(
    hrp: HRPStyle,
    ordenar_por_clusters: bool = True,
    file_path: Optional[str] = None,
) -> go.Figure:
    """
    Backward-compatible wrapper for the original Spanish helper name.
    """

    return plot_distance_matrix(hrp, reorder_by_clusters=ordenar_por_clusters, file_path=file_path)


__all__ = ["plot_distance_matrix", "matriz_distancias"]
