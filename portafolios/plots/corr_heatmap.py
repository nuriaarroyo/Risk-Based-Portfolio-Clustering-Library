from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import PortfolioUniverse


def plot_portfolio_heatmap(
    portfolio: "PortfolioUniverse",
    kind: Literal["correlation", "covariance"] = "correlation",
    round_decimals: int = 2,
) -> None:
    """
    Plot the portfolio correlation or covariance matrix as a Plotly heatmap.

    Uses only objects already present on `PortfolioUniverse`:
    - `portfolio.correlation`
    - `portfolio.covariance`
    - `portfolio.asset_log_returns` or `asset_returns` as a fallback
    - `portfolio.info["constructor_display_name"]` for the title
    """

    if kind == "correlation":
        matrix = portfolio.correlation
        if matrix is None:
            if portfolio.asset_log_returns is not None:
                matrix = portfolio.asset_log_returns.corr()
            elif portfolio.asset_returns is not None:
                matrix = portfolio.asset_returns.corr()
            else:
                print("No returns are available to compute the correlation matrix.")
                return
        title_kind = "Correlation"
    elif kind == "covariance":
        matrix = portfolio.covariance
        if matrix is None:
            if portfolio.asset_log_returns is not None:
                matrix = portfolio.asset_log_returns.cov()
            elif portfolio.asset_returns is not None:
                matrix = portfolio.asset_returns.cov()
            else:
                print("No returns are available to compute the covariance matrix.")
                return
        title_kind = "Covariance"
    else:
        raise ValueError("`kind` must be 'correlation' or 'covariance'.")

    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        print("The selected matrix is empty or is not a DataFrame.")
        return

    matrix = matrix.loc[matrix.index, matrix.index]
    values = matrix.values.astype(float)
    tickers = matrix.index.tolist()

    if kind == "correlation":
        zmin, zmax = -1.0, 1.0
    else:
        max_abs = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        zmin, zmax = -max_abs, max_abs

    text = np.vectorize(lambda value: f"{value:.{round_decimals}f}")(values)

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
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
        title=f"{title_kind} Heatmap - {constructor_name}",
        xaxis_title="Assets",
        yaxis_title="Assets",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=900,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    safe_constructor = str(constructor_name).replace(" ", "_").replace("/", "_")
    suffix = "corr" if kind == "correlation" else "cov"
    plots_dir = Path(getattr(portfolio, "plots_dir", Path.cwd() / "outputs" / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_path = plots_dir / f"heatmap_{suffix}_{safe_constructor}.html"
    fig.write_html(str(out_path))

    print(f"Saved {title_kind.lower()} heatmap to:\n{out_path}")
    fig.show()


def corr_heatmap_portfolio(
    portfolio: "PortfolioUniverse",
    kind: Literal["correlation", "covariance"] = "correlation",
    round_decimals: int = 2,
) -> None:
    """
    Backward-compatible wrapper for the original helper name.
    """

    return plot_portfolio_heatmap(portfolio, kind=kind, round_decimals=round_decimals)


__all__ = ["plot_portfolio_heatmap", "corr_heatmap_portfolio"]
