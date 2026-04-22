from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

from portafolios.constructores.hrp_style.hrp_core import HRPStyle


def plot_distance_histogram(
    hrp: HRPStyle,
    bins: int = 100,
    file_path: Optional[str] = "hrp_distance_histogram.html",
):
    """
    Plot a histogram of pairwise distances (upper triangle without the diagonal).
    """

    if not hasattr(hrp, "last_dist"):
        raise RuntimeError("This HRPStyle instance does not expose 'last_dist'. Run the HRP construction step first.")

    distance_matrix: pd.DataFrame = hrp.last_dist

    upper_triangle = np.triu_indices_from(distance_matrix.values, k=1)
    distance_values = distance_matrix.values[upper_triangle].ravel()

    fig = px.histogram(
        x=distance_values,
        nbins=bins,
        labels={"x": "Distance", "y": "Frequency"},
        title="Asset Distance Histogram",
    )

    if file_path is not None:
        fig.write_html(file_path)

    fig.show()
    return fig


def histograma_distancias(
    hrp: HRPStyle,
    bins: int = 100,
    file_path: Optional[str] = "hrp_hist_distancias_deprado.html",
):
    """
    Backward-compatible wrapper for the original Spanish helper name.
    """

    return plot_distance_histogram(hrp, bins=bins, file_path=file_path)


__all__ = ["plot_distance_histogram", "histograma_distancias"]
