from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import PortfolioUniverse


def plot_portfolio_bar(
    portfolio: "PortfolioUniverse",
    weights: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Plot portfolio weights as a bar chart.

    Uses:
    - `portfolio.asset_returns.columns` as the asset universe
    - `portfolio.weights` as the default weights
    - `portfolio.info["constructor_display_name"]` for the title, when available

    If `weights` is None, uses `portfolio.weights`.
    Aligns to `asset_returns.columns`; missing Series entries are filled with 0.
    Normalizes if the weights do not sum to 1.
    Can filter very small weights with `min_weight`.
    Saves HTML to the configured universe output directory or `./outputs/plots/`.
    """

    if portfolio.asset_returns is None:
        print("No asset returns are available. Run the data preparation step first.")
        return

    tickers = list(portfolio.asset_returns.columns)
    if not tickers:
        print("No tickers are available in asset_returns.")
        return

    if weights is None:
        if getattr(portfolio, "weights", None) is None:
            print("No weights are available on the portfolio. Run the portfolio construction step first or pass `weights`.")
            return

        if isinstance(portfolio.weights, pd.Series):
            aligned_weights = portfolio.weights.reindex(tickers).fillna(0.0).values.astype(float)
        else:
            aligned_weights = np.asarray(portfolio.weights, dtype=float)
            if len(aligned_weights) != len(tickers):
                raise ValueError(
                    f"Weight length ({len(aligned_weights)}) does not match the number of assets ({len(tickers)})."
                )
    elif isinstance(weights, pd.Series):
        aligned_weights = weights.reindex(tickers).fillna(0.0).values.astype(float)
    else:
        aligned_weights = np.asarray(weights, dtype=float)
        if len(aligned_weights) != len(tickers):
            raise ValueError(
                f"Weight length ({len(aligned_weights)}) does not match the number of assets ({len(tickers)})."
            )

    if not np.isfinite(aligned_weights).all():
        raise ValueError("Weights contain NaN or infinite values.")

    aligned_weights[aligned_weights < 0] = np.maximum(aligned_weights[aligned_weights < 0], -1e-12)

    weight_sum = aligned_weights.sum()
    if weight_sum == 0:
        print("Weights sum to zero; there is nothing to plot.")
        return
    if not np.isclose(weight_sum, 1.0, atol=1e-8):
        aligned_weights = aligned_weights / weight_sum

    mask = aligned_weights > min_weight
    plot_weights = aligned_weights[mask]
    plot_tickers = [ticker for ticker, keep in zip(tickers, mask) if keep]

    if len(plot_weights) == 0:
        print(f"All weights are <= {min_weight:.2%}; there is nothing to plot.")
        return

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_tickers,
                y=plot_weights,
                text=[f"{weight:.1%}" for weight in plot_weights],
                textposition="auto",
            )
        ]
    )

    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )

    fig.update_layout(
        title=f"Portfolio Weight Distribution - {constructor_name}",
        xaxis_title="Asset",
        yaxis_title="Weight",
        width=1400,
        height=700,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    safe_name = (
        str(constructor_name)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    plots_dir = Path(getattr(portfolio, "plots_dir", Path.cwd() / "outputs" / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_path = plots_dir / f"portfolio_bar_{safe_name}.html"
    fig.write_html(str(out_path))
    print(f"Saved bar chart to: {out_path}")
    fig.show()


def barras_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Backward-compatible wrapper for the original Spanish helper name.
    """

    return plot_portfolio_bar(portfolio, weights=pesos, min_weight=min_weight)


__all__ = ["plot_portfolio_bar", "barras_portfolio"]
