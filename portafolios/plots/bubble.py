from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.universe import PortfolioUniverse


def bubbleplot_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Plotly bubble chart of expected return vs volatility by asset.

    Uses:
    - `portfolio.asset_returns` (daily simple returns)
    - `portfolio.tickers`
    - `portfolio.weights`
    - `portfolio.info["constructor_display_name"]` for the title

    Bubble size is proportional to portfolio weight.
    """

    # Base validation.
    if portfolio.asset_returns is None:
        print("Llama primero a preparar_datos().")
        return

    tickers = list(portfolio.asset_returns.columns)
    if len(tickers) == 0:
        print("No hay tickers en asset_returns.")
        return

    # Per-asset expected return and volatility.
    rets = portfolio.asset_returns.dropna()
    if rets.empty:
        print("No hay retornos para ER/VOL.")
        return

    er = rets.mean()
    vol = rets.std()

    # Get and align weights.
    if pesos is None:
        if portfolio.weights is None:
            print("Llama a construir() antes o pasa pesos explicitos.")
            return

        if isinstance(portfolio.weights, pd.Series):
            w = portfolio.weights.reindex(tickers).fillna(0.0).astype(float)
        else:
            arr = np.asarray(portfolio.weights, dtype=float)
            if len(arr) != len(tickers):
                raise ValueError("El tamano de weights no coincide con los activos.")
            w = pd.Series(arr, index=tickers)

    elif isinstance(pesos, pd.Series):
        w = pesos.reindex(tickers).fillna(0.0).astype(float)

    else:
        arr = np.asarray(pesos, dtype=float)
        if len(arr) != len(tickers):
            raise ValueError("El tamano de pesos no coincide con los activos.")
        w = pd.Series(arr, index=tickers)

    # Clean and normalize.
    w[w < 0] = np.maximum(w[w < 0], -1e-12)
    ssum = w.sum()
    if not np.isclose(ssum, 1.0, atol=1e-8):
        w = w / ssum

    # Filter by minimum weight.
    mask = w > min_weight
    if not mask.any():
        print(f"Todos los pesos <= {min_weight:.4f}. Nada que graficar.")
        return

    w_plot = w[mask].values
    er_plot = er[mask].values
    vol_plot = vol[mask].values
    labels = w.index[mask].tolist()

    # Bubble sizes.
    n_pts = len(w_plot)
    base = 1800 if n_pts <= 25 else (1400 if n_pts <= 60 else 1000)
    sizes = np.maximum(8, np.sqrt(w_plot) * base)

    # Plotly figure.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=er_plot,
            y=vol_plot,
            mode="markers+text" if n_pts <= 25 else "markers",
            marker=dict(size=sizes, opacity=0.7, line=dict(color="black", width=1)),
            text=labels if n_pts <= 25 else None,
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>ER: %{x:.4f}<br>VOL: %{y:.4f}<br>Peso: %{customdata:.2%}<extra></extra>",
            customdata=w_plot,
            name="Activos",
        )
    )

    constructor = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )

    fig.update_layout(
        title=f"Bubble Plot: ER vs VOL - {constructor}",
        xaxis_title="Expected Return (diario)",
        yaxis_title="Volatility (diaria)",
        width=1200,
        height=900,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Save output.
    safe = str(constructor).replace(" ", "_").replace("/", "_")
    plots_dir = Path(getattr(portfolio, "plots_dir", Path.cwd() / "outputs" / "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"bubble_plot_{safe}.html"
    fig.write_html(str(out_path))

    print(f"Bubble plot guardado en:\n{out_path}")
    fig.show()
