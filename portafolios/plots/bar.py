# portafolios/plots/barpy.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ..core.portafolio import Portfolio


def barras_portfolio(
    portfolio: "Portfolio",
    pesos: Optional[Union[Sequence[float], pd.Series]] = None,
    min_weight: float = 0.0,
) -> None:
    """
    Grafica barras de los pesos del portafolio.

    Usa:
    - portfolio.asset_returns.columns  como universo de activos
    - portfolio.weights                como pesos por defecto
    - portfolio.info["constructor"]    para el título (si existe)

    - Si `pesos` es None, usa `portfolio.weights`.
    - Alinea a asset_returns.columns; si faltan pesos en una Series, rellena con 0.
    - Normaliza si la suma != 1.
    - Puede filtrar pesos muy pequeños (min_weight).
    - Guarda HTML en ./plots/ y hace fig.show().
    """

    # --- universo de activos ---
    if portfolio.asset_returns is None:
        print("No hay retornos de activos. Llama primero a preparar_datos().")
        return

    tickers = list(portfolio.asset_returns.columns)
    if not tickers:
        print("No hay tickers en asset_returns.")
        return

    # --- obtener / alinear pesos ---
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
                    f"Length of weights ({len(w)}) no coincide con número de activos ({len(tickers)})."
                )
    elif isinstance(pesos, pd.Series):
        s = pesos.reindex(tickers).fillna(0.0)
        w = s.values.astype(float)
    else:
        w = np.asarray(pesos, dtype=float)
        if len(w) != len(tickers):
            raise ValueError(
                f"Length of weights ({len(w)}) no coincide con número de activos ({len(tickers)})."
            )

    # --- limpieza / normalización ---
    if not np.isfinite(w).all():
        raise ValueError("Los pesos contienen NaN o Inf.")

    w[w < 0] = np.maximum(w[w < 0], -1e-12)

    ssum = w.sum()
    if ssum == 0:
        print("Suma de pesos = 0; nada que graficar.")
        return
    if not np.isclose(ssum, 1.0, atol=1e-8):
        w = w / ssum

    # --- filtrar pesos pequeños ---
    mask = w > min_weight
    w_plot = w[mask]
    t_plot = [t for t, m in zip(tickers, mask) if m]

    if len(w_plot) == 0:
        print(f"Todos los pesos son ≤ {min_weight:.2%}; nada que graficar.")
        return

    # --- gráfico ---
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

    constructor_name = portfolio.info.get("constructor", "Portafolio")

    fig.update_layout(
        title=f"Distribución de Pesos del {constructor_name}",
        xaxis_title="Activos",
        yaxis_title="Peso",
        width=1400,
        height=700,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # --- guardar y mostrar ---
    safe_name = str(constructor_name).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")

    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(exist_ok=True)

    out_path = plots_dir / f"portfolio_barras_{safe_name}.html"
    fig.write_html(str(out_path))
    print(f"Gráfica de barras guardada en: {out_path}")

    # para nb / navegador
    fig.show()
