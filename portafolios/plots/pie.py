from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.universe import PortfolioUniverse


def pastel_portfolio(
    portfolio: "PortfolioUniverse",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 1e-3,
) -> None:
    """
    Plot a donut chart of `PortfolioUniverse` weights.

    Uses only:
    - `portfolio.asset_returns.columns` (asset universe)
    - `portfolio.weights` (constructed weights)
    - `portfolio.info["constructor_display_name"]` for the title, when available

    If `pesos` is None, uses `portfolio.weights`.
    Filters very small weights with `min_weight`.
    Saves the HTML to the configured output directory or, if missing,
    to `./outputs/plots/`.
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
            print("No hay pesos en el objeto. Llama a construir(...) primero o pasa 'pesos'.")
            return

        # `weights` is expected to be a Series indexed by tickers.
        if isinstance(portfolio.weights, pd.Series):
            w = portfolio.weights.reindex(tickers).fillna(0.0).values.astype(float)
        else:
            # Guard against constructors that return arrays; assume column order.
            w = np.asarray(portfolio.weights, dtype=float)
            if len(w) != len(tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) no coincide con numero de activos ({len(tickers)})."
                )
    elif isinstance(pesos, pd.Series):
        # Align with the `asset_returns` universe.
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

    # Trim tiny numerical negatives.
    w[w < 0] = np.maximum(w[w < 0], -1e-12)

    ssum = w.sum()
    if ssum == 0:
        print("Suma de pesos = 0; nada que graficar.")
        return

    if not np.isclose(ssum, 1.0, atol=1e-8):
        w = w / ssum

    # Filter small weights.
    mask = w > min_weight
    pesos_filtrados = w[mask].tolist()
    tickers_filtrados = [ticker for ticker, keep in zip(tickers, mask) if keep]

    if len(pesos_filtrados) == 0:
        print(f"Todos los pesos son <= {min_weight:.2%}; nada que graficar.")
        return

    # Adapt sizing and text to the total asset count.
    n_activos = len(tickers)
    if n_activos <= 20:
        width, height = 1200, 800
        text_size = 12
        legend_size = 10
    elif n_activos <= 40:
        width, height = 1400, 900
        text_size = 11
        legend_size = 9
    else:
        width, height = 1600, 1000
        text_size = 10
        legend_size = 8

    # Detect near-equal-weight allocations.
    es_naive = False
    weight_array = np.array(pesos_filtrados, dtype=float)
    if weight_array.mean() > 0 and (weight_array.std() / weight_array.mean()) < 0.02 and len(weight_array) > 10:
        es_naive = True

    # Donut-hole size.
    if len(pesos_filtrados) < len(tickers) * 0.5:
        hole_size = 0.1
    elif es_naive:
        hole_size = 0.2
    else:
        hole_size = 0.3

    # Colors and pull effect.
    if es_naive:
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        reps = (len(tickers_filtrados) // len(base_colors)) + 1
        colors = (base_colors * reps)[: len(tickers_filtrados)]
        pull_effect = [0.05] * len(tickers_filtrados)
    else:
        colors = None
        pull_effect = [0.1 if weight > 0.05 else 0 for weight in pesos_filtrados]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=tickers_filtrados,
                values=pesos_filtrados,
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

    # Build a readable title using `info["constructor"]` when available.
    constructor_name = portfolio.info.get(
        "constructor_display_name",
        portfolio.info.get("constructor", "Portfolio"),
    )
    activos_excluidos = len(tickers) - len(tickers_filtrados)

    if activos_excluidos > 0:
        title_text = (
            f"Pesos del {constructor_name}"
            f"<br><sub>Se muestran {len(tickers_filtrados)} de {len(tickers)} activos "
            f"(excluidos {activos_excluidos} con peso <= {min_weight:.2%})</sub>"
        )
    elif es_naive:
        title_text = (
            f"Pesos del {constructor_name}"
            f"<br><sub>Portfolio equiponderado: {len(tickers_filtrados)} activos "
            f"~ {pesos_filtrados[0]:.1%} cada uno</sub>"
        )
    else:
        title_text = f"Pesos del {constructor_name}"

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

    # Save to the configured output directory.
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

    out_path = plots_dir / f"portfolio_pastel_{safe_name}.html"
    fig.write_html(str(out_path))
    fig.show()

    print(f"Grafica de pastel guardada en: {out_path}")
