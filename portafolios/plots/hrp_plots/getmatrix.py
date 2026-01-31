from __future__ import annotations
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def get_distmat(
    hrp: HRPStyle,
    ordenar_por_clusters: bool = True,
    file_path: Optional[str] = None,
) -> go.Figure:
    """
    Heatmap Plotly de la matriz de distancias usada por HRP.

    - Si ordenar_por_clusters=True, reordena filas/columnas según los clusters.
    - Si file_path se da, guarda el plot como HTML en esa ruta.
    """
    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("Este HRPStyle no tiene 'last_dist'. ¿Ya corriste p.construir(hrp)?")

    dist: pd.DataFrame = hrp.last_dist.copy()

    return dist
