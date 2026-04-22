from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import PortfolioUniverse


def plot_portfolio_bubble(
    portfolio: "PortfolioUniverse",
    weights: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Plot a bubble chart of expected return versus volatility by asset.

    Uses:
    - `portfolio.asset_returns` for daily simple returns
    - `portfolio.weights` as the default weights
    - `portfolio.info["constructor_display_name"]` for the title

    Bubble size is proportional to the portfolio weight.
    """

    if portfolio.asset_returns is None:
        print("No asset returns are available. Run the data preparation step first.")
        return

    tickers = list(portfolio.asset_returns.columns)
    if not tickers:
        print("No tickers are available in asset_returns.")
        return

    returns = portfolio.asset_returns.dropna()
    if returns.empty:
        print("No return history is available to compute expected return and volatility.")
        return

    expected_return = returns.mean()
    volatility = returns.std()

    if weights is None:
        if portfolio.weights is None:
            print("Run the portfolio construction step first or pass explicit `weights`.")
            return

        if isinstance(portfolio.weights, pd.Series):
            aligned_weights = portfolio.weights.reindex(tickers).fillna(0.0).astype(float)
        else:
            aligned_array = np.asarray(portfolio.weights, dtype=float)
            if len(aligned_array) != len(tickers):
                raise ValueError("Weight length does not match the number of assets.")
            aligned_weights = pd.Series(aligned_array, index=tickers)
    elif isinstance(weights, pd.Series):
        aligned_weights = weights.reindex(tickers).fillna(0.0).astype(float)
    else:
        aligned_array = np.asarray(weights, dtype=float)
        if len(aligned_array) != len(tickers):
            raise ValueError("Weight length does not match the number of assets.")
        aligned_weights = pd.Series(aligned_array, index=tickers)

    aligned_weights[aligned_weights < 0] = np.maximum(aligned_weights[aligned_weights < 0], -1e-12)
    weight_sum = float(aligned_weights.sum())
    if weight_sum == 0:
        print("Weights sum to zero; there is nothing to plot.")
        return
    if not np.isclose(weight_sum, 1.0, atol=1e-8):
        aligned_weights = aligned_weights / weight_sum

    mask = aligned_weights > min_weight
    if not mask.any():
        print(f"All weights are <= {min_weight:.4f}; there is nothing to plot.")
        return

    plot_weights = aligned_weights[mask].values
    plot_returns = expected_return[mask].values
    plot_volatility = volatility[mask].values
    labels = aligned_weights.index[mask].tolist()

    n_points = len(plot_weights)
    base_scale = 1800 if n_points <= 25 else (1400 if n_points <= 60 else 1000)
    bubble_sizes = np.maximum(8, np.sqrt(plot_weights) * base_scale)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_returns,
            y=plot_volatility,
            mode="markers+text" if n_points <= 25 else "markers",
            marker=dict(size=bubble_sizes, opacity=0.7, line=dict(color="black", width=1)),
            text=labels if n_points <= 25 else None,
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>ER: %{x:.4f}<br>VOL: %{y:.4f}<br>Weight: %{customdata:.2%}<extra></extra>",
            customdata=plot_weights,
            name="Assets",
        )
    )

    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )

    fig.update_layout(
        title=f"Bubble Plot: Expected Return vs Volatility - {constructor_name}",
        xaxis_title="Expected Return (daily)",
        yaxis_title="Volatility (daily)",
        width=1200,
        height=900,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    safe_name = str(constructor_name).replace(" ", "_").replace("/", "_")
    plots_dir = Path(getattr(portfolio, "plots_dir", Path.cwd() / "outputs" / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"bubble_plot_{safe_name}.html"
    fig.write_html(str(out_path))

    print(f"Saved bubble plot to:\n{out_path}")
    fig.show()


def bubbleplot_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Backward-compatible wrapper for the original Spanish helper name.
    """

    return plot_portfolio_bubble(portfolio, weights=pesos, min_weight=min_weight)


__all__ = ["plot_portfolio_bubble", "bubbleplot_portfolio"]
