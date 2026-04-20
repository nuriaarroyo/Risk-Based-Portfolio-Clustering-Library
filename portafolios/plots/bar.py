from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.universe import PortfolioUniverse


def barras_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Plot portfolio weights as a bar chart.

    Uses:
    - `portfolio.asset_returns.columns` as the asset universe
    - `portfolio.weights` as the default weights
    - `portfolio.info["constructor_display_name"]` for the title, when available

    - If `pesos` is None, uses `portfolio.weights`.
    - Aligns to `asset_returns.columns`; missing Series entries are filled with 0.
    - Normalizes if the weights do not sum to 1.
    - Can filter very small weights with `min_weight`.
    - Saves HTML to the configured universe output directory or `./outputs/plots/`.
    """

    # Asset universe.
    if portfolio.asset_returns is None:
        print("No hay retornos de activos. Llama primero a preparar_datos().")
        return

    tickers = list(portfolio.asset_returns.columns)
    if not tickers:
        print("No hay tickers en asset_returns.")
        return

    # Get and align weights.
    if pesos is None:
        if getattr(portfolio, "weights", None) is None:
            print("No hay pesos en el objeto. Llama a construir(...) o pasa `pesos`.")
            return

        if isinstance(portfolio.weights, pd.Series):
            w = portfolio.weights.reindex(tickers).fillna(0.0).values.astype(float)
        else:
            w = np.asarray(portfolio.weights, dtype=float)
            if len(w) != len(tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) no coincide con numero de activos ({len(tickers)})."
                )
    elif isinstance(pesos, pd.Series):
        s = pesos.reindex(tickers).fillna(0.0)
        w = s.values.astype(float)
    else:
        w = np.asarray(pesos, dtype=float)
        if len(w) != len(tickers):
            raise ValueError(
                f"Length of weights ({len(w)}) no coincide con numero de activos ({len(tickers)})."
            )

    # Clean and normalize.
    if not np.isfinite(w).all():
        raise ValueError("Los pesos contienen NaN o Inf.")

    w[w < 0] = np.maximum(w[w < 0], -1e-12)

    ssum = w.sum()
    if ssum == 0:
        print("Suma de pesos = 0; nada que graficar.")
        return
    if not np.isclose(ssum, 1.0, atol=1e-8):
        w = w / ssum

    # Filter small weights.
    mask = w > min_weight
    w_plot = w[mask]
    t_plot = [t for t, m in zip(tickers, mask) if m]

    if len(w_plot) == 0:
        print(f"Todos los pesos son <= {min_weight:.2%}; nada que graficar.")
        return

    # Plot.
    fig = go.Figure(
        data=[
            go.Bar(
                x=t_plot,
                y=w_plot,
                text=[f"{p:.1%}" for p in w_plot],
                textposition="auto",
            )
        ]
    )

    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )

    fig.update_layout(
        title=f"Distribucion de Pesos del {constructor_name}",
        xaxis_title="Activos",
        yaxis_title="Peso",
        width=1400,
        height=700,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Save and display.
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

    out_path = plots_dir / f"portfolio_barras_{safe_name}.html"
    fig.write_html(str(out_path))
    print(f"Grafica de barras guardada en: {out_path}")

    # Show for notebook/browser workflows.
    fig.show()
