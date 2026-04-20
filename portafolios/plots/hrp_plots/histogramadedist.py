from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def histograma_distancias(
    hrp: HRPStyle,
    bins: int = 100,
    file_path: Optional[str] = "hrp_hist_distancias_deprado.html",
):
    """
    Plotly histogram of pairwise distances (upper triangle without the diagonal).
    """
    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("Este HRPStyle no tiene 'last_dist'. ¿Ya corriste p.construir(hrp)?")

    dist: pd.DataFrame = hrp.last_dist

    # Upper triangle without the diagonal.
    triu = np.triu_indices_from(dist.values, k=1)
    vals = dist.values[triu].ravel()

    fig = px.histogram(
        x=vals,
        nbins=bins,
        labels={"x": "Distancia ", "y": "Frecuencia"},
        title="Histograma de distancias entre activos",
    )

    if file_path is not None:
        fig.write_html(file_path)

    fig.show()

    return fig
