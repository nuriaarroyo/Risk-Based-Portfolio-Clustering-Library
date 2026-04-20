from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def matriz_distancias(
    hrp: HRPStyle,
    ordenar_por_clusters: bool = True,
    file_path: Optional[str] = None,
) -> go.Figure:
    """
    Plotly heatmap of the distance matrix used by HRP.

    hrp: `HRPStyle` object already used in `p.construir(hrp)`
    """
    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("Este HRPStyle no tiene 'last_dist'. ¿Ya corriste p.construir(hrp)?")

    dist: pd.DataFrame = hrp.last_dist.copy()

    # Reorder by clusters when available.
    if ordenar_por_clusters and hasattr(hrp, "last_clusters"):
        ordered = [asset for cluster in hrp.last_clusters for asset in cluster]
        ordered = [asset for asset in ordered if asset in dist.index]
        dist = dist.loc[ordered, ordered]

    fig = go.Figure(
        data=go.Heatmap(
            z=dist.values,
            x=list(dist.columns),
            y=list(dist.index),
            colorscale="RdBu",
            colorbar=dict(title="Distancia"),
        )
    )

    fig.update_layout(
        title="Matriz de distancias de De Prado ",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        width=800,
        height=700,
    )

    if file_path is not None:
        fig.write_html(file_path)

    fig.show()
    return fig
