# portafolios/constructores/markowitz.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseConstructor


class Markowitz(BaseConstructor):
    """
    Constructor Markowitz que maximiza el Sharpe ratio histórico.

    Nota: PortfolioUniverse.construir le pasa `asset_returns` (retornos simples).
    Aquí decidimos SI LOS USAMOS tal cual o los convertimos a log,
    según el parámetro `ret_kind`.
    """

    method_id = "markowitz_max_sharpe"
    display_name = "Markowitz Max Sharpe"

    def __init__(self, rf_per_period: float = 0.0):
        self.rf_per_period = rf_per_period

    def optimizar(
        self,
        returns: pd.DataFrame,
        *,
        ret_kind: str = "log",   # <- aquí pedimos el tipo de retorno a usar
        allow_short: bool = False,
        bounds: Any = None,
        **kwargs,
    ) -> tuple[pd.Series, dict[str, Any]]:

        if returns is None or returns.empty:
            raise ValueError("El DataFrame de retornos está vacío o es None.")

        # returns que vienen de PortfolioUniverse son retornos simples.
        rets = returns.dropna(axis=0, how="any")
        if rets.empty:
            raise ValueError("No hay filas sin NaN en los retornos.")

        # --- elegir qué tipo de retornos usar en la optimización --- 
        # ret_kind = "log"  -> usamos log(1 + r_simple)
        # ret_kind = "simple" -> usamos r_simple tal cual
        if ret_kind == "log":
            rets_used = np.log1p(rets)   # log(1+r)
        elif ret_kind in ("simple", "normal", "pct"):
            rets_used = rets
        else:
            raise ValueError("ret_kind debe ser 'log' o 'simple'.")

        tickers = list(rets_used.columns)
        n = len(tickers)

        rf = float(kwargs.get("rf_per_period", self.rf_per_period))

        mean_returns = rets_used.mean()
        cov_matrix   = rets_used.cov()

        # --- bounds ---
        if bounds is not None:
            bounds_tuple = tuple(bounds)
            if len(bounds_tuple) != n:
                raise ValueError("bounds debe tener longitud = número de activos.")
        else:
            if allow_short:
                bounds_tuple = tuple((-1.0, 1.0) for _ in range(n))
            else:
                bounds_tuple = tuple((0.0, 1.0) for _ in range(n))

        # restricción suma de pesos = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        x0 = np.ones(n) / n

        def neg_sharpe(w: np.ndarray) -> float:
            w = np.asarray(w, dtype=float)
            port_ret = float(w @ mean_returns.values)
            port_var = float(w @ cov_matrix.values @ w)
            port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
            if port_vol == 0.0:
                return 1e6
            return -(port_ret - rf) / port_vol

        result = minimize(
            fun=neg_sharpe,
            x0=x0,
            method="SLSQP",
            bounds=bounds_tuple,
            constraints=constraints,
        )

        if not result.success:
            w_opt = pd.Series(x0, index=tickers)
            meta = {
                "n_assets": n,
                "success": False,
                "message": result.message,
                "rf_per_period": rf,
                "objective": None,
                "allow_short": allow_short,
                "ret_kind_used": ret_kind,
            }
            return w_opt, meta

        w_opt = pd.Series(result.x, index=tickers)
        ssum = float(w_opt.sum())
        if not np.isclose(ssum, 1.0, atol=1e-6):
            w_opt = w_opt / ssum

        meta = {
            "n_assets": n,
            "success": True,
            "message": result.message,
            "rf_per_period": rf,
            "objective": -float(result.fun),  # Sharpe max alcanzado
            "allow_short": allow_short,
            "ret_kind_used": ret_kind,
        }

        return w_opt, meta
