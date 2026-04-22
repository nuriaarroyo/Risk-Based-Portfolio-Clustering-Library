from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import PortfolioUniverse


def plot_portfolio_pie(
    portfolio: "PortfolioUniverse",
    weights: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 1e-3,
) -> None:
    """
    Plot a donut chart of portfolio weights.

    Uses only:
    - `portfolio.asset_returns.columns` as the asset universe
    - `portfolio.weights` as the default weights
    - `portfolio.info["constructor_display_name"]` for the title, when available

    If `weights` is None, uses `portfolio.weights`.
    Filters very small weights with `min_weight`.
    Saves the HTML to the configured output directory or, if missing,
    to `./outputs/plots/`.
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
    filtered_weights = aligned_weights[mask].tolist()
    filtered_tickers = [ticker for ticker, keep in zip(tickers, mask) if keep]

    if len(filtered_weights) == 0:
        print(f"All weights are <= {min_weight:.2%}; there is nothing to plot.")
        return

    asset_count = len(tickers)
    if asset_count <= 20:
        width, height = 1200, 800
        text_size = 12
        legend_size = 10
    elif asset_count <= 40:
        width, height = 1400, 900
        text_size = 11
        legend_size = 9
    else:
        width, height = 1600, 1000
        text_size = 10
        legend_size = 8

    looks_equal_weight = False
    weight_array = np.array(filtered_weights, dtype=float)
    if weight_array.mean() > 0 and (weight_array.std() / weight_array.mean()) < 0.02 and len(weight_array) > 10:
        looks_equal_weight = True

    if len(filtered_weights) < len(tickers) * 0.5:
        hole_size = 0.1
    elif looks_equal_weight:
        hole_size = 0.2
    else:
        hole_size = 0.3

    if looks_equal_weight:
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        repetitions = (len(filtered_tickers) // len(base_colors)) + 1
        colors = (base_colors * repetitions)[: len(filtered_tickers)]
        pull_effect = [0.05] * len(filtered_tickers)
    else:
        colors = None
        pull_effect = [0.1 if weight > 0.05 else 0 for weight in filtered_weights]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=filtered_tickers,
                values=filtered_weights,
                hole=hole_size,
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=text_size),
                marker=dict(line=dict(color="white", width=1), colors=colors),
                rotation=45,
                pull=pull_effect,
            )
        ]
    )

    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )
    excluded_assets = len(tickers) - len(filtered_tickers)

    if excluded_assets > 0:
        title_text = (
            f"Portfolio Weights - {constructor_name}"
            f"<br><sub>Showing {len(filtered_tickers)} of {len(tickers)} assets "
            f"(excluded {excluded_assets} with weight <= {min_weight:.2%})</sub>"
        )
    elif looks_equal_weight:
        title_text = (
            f"Portfolio Weights - {constructor_name}"
            f"<br><sub>Equal-weight allocation: {len(filtered_tickers)} assets "
            f"~ {filtered_weights[0]:.1%} each</sub>"
        )
    else:
        title_text = f"Portfolio Weights - {constructor_name}"

    fig.update_layout(
        title=title_text,
        showlegend=True,
        width=width,
        height=height,
        title_font_size=20,
        font=dict(size=legend_size),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=legend_size),
        ),
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

    out_path = plots_dir / f"portfolio_pie_{safe_name}.html"
    fig.write_html(str(out_path))
    fig.show()

    print(f"Saved pie chart to: {out_path}")


def pastel_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 1e-3,
) -> None:
    """
    Backward-compatible wrapper for the original Spanish helper name.
    """

    return plot_portfolio_pie(portfolio, weights=pesos, min_weight=min_weight)


__all__ = ["plot_portfolio_pie", "pastel_portfolio"]
