from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import PortfolioUniverse


def corr_heatmap_portfolio(
    portfolio: "PortfolioUniverse",
    kind: Literal["correlation", "covariance"] = "correlation",
    round_decimals: int = 2,
) -> None:
    """
    Plotly heatmap of the portfolio correlation or covariance matrix.

    Uses only objects already present on `PortfolioUniverse`:
    - `portfolio.correlation`
    - `portfolio.covariance`
    - `portfolio.asset_log_returns` or `asset_returns` as fallback
    - `portfolio.info["constructor_display_name"]` for the title

    Parameters
    ----------
    kind : {"correlation", "covariance"}
        - "correlation": use the correlation matrix.
        - "covariance": use the covariance matrix.
    round_decimals : int
        Number of decimals shown in annotations.
    """

    # Choose the base matrix.
    if kind == "correlation":
        mat = portfolio.correlation
        if mat is None:
            # Fallback in case it was not computed during `preparar_datos`.
            if portfolio.asset_log_returns is not None:
                mat = portfolio.asset_log_returns.corr()
            elif portfolio.asset_returns is not None:
                mat = portfolio.asset_returns.corr()
            else:
                print("No hay retornos para calcular la correlacion.")
                return
        title_kind = "Correlacion"
    elif kind == "covariance":
        mat = portfolio.covariance
        if mat is None:
            if portfolio.asset_log_returns is not None:
                mat = portfolio.asset_log_returns.cov()
            elif portfolio.asset_returns is not None:
                mat = portfolio.asset_returns.cov()
            else:
                print("No hay retornos para calcular la covarianza.")
                return
        title_kind = "Covarianza"
    else:
        raise ValueError("kind debe ser 'correlation' o 'covariance'.")

    if not isinstance(mat, pd.DataFrame) or mat.empty:
        print("La matriz seleccionada esta vacia o no es un DataFrame.")
        return

    # Ensure rows and columns are aligned in the same order.
    mat = mat.loc[mat.index, mat.index]

    # Numeric values.
    z = mat.values.astype(float)
    tickers = mat.index.tolist()

    # Color limits.
    if kind == "correlation":
        zmin, zmax = -1.0, 1.0
    else:
        # Covariance can be highly unbalanced, so use symmetric limits.
        max_abs = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
        zmin, zmax = -max_abs, max_abs

    # Rounded annotation text.
    text = np.vectorize(lambda v: f"{v:.{round_decimals}f}")(z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            colorscale="RdBu",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=title_kind),
            text=text,
            hoverinfo="text",
        )
    )

    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )

    fig.update_layout(
        title=f"Heatmap de {title_kind} - {constructor_name}",
        xaxis_title="Activos",
        yaxis_title="Activos",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=900,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Save output.
    safe_constructor = str(constructor_name).replace(" ", "_").replace("/", "_")
    suffix = "corr" if kind == "correlation" else "cov"
    plots_dir = Path(getattr(portfolio, "plots_dir", Path.cwd() / "outputs" / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_path = plots_dir / f"heatmap_{suffix}_{safe_constructor}.html"
    fig.write_html(str(out_path))

    print(f"Heatmap de {title_kind} guardado en:\n{out_path}")
    fig.show()
