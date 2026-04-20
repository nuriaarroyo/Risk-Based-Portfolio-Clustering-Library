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
    Return the HRP distance matrix used by the last run.

    - If `ordenar_por_clusters=True`, rows and columns are reordered by cluster.
    - If `file_path` is provided, the plot can be saved as HTML at that path.
    """
    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("Este HRPStyle no tiene 'last_dist'. ¿Ya corriste p.construir(hrp)?")

    dist: pd.DataFrame = hrp.last_dist.copy()

    return dist
