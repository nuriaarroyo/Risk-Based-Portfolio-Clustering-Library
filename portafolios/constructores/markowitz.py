from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseConstructor


class Markowitz(BaseConstructor):
    """
    Markowitz constructor that maximizes the historical Sharpe ratio.

    Note: `PortfolioUniverse.construir` passes `asset_returns` (simple returns).
    They are used as-is by default to keep behavior consistent across the
    library. They can optionally be converted to log returns with
    `ret_kind="log"`.
    """

    method_id = "markowitz_max_sharpe"
    display_name = "Markowitz Max Sharpe"

    def __init__(self, rf_per_period: float = 0.0):
        self.rf_per_period = rf_per_period

    def optimizar(
        self,
        returns: pd.DataFrame,
        *,
        ret_kind: str = "simple",
        allow_short: bool = False,
        bounds: Any = None,
        **kwargs,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if returns is None or returns.empty:
            raise ValueError("El DataFrame de retornos esta vacio o es None.")

        # Returns coming from PortfolioUniverse are simple returns.
        rets = returns.dropna(axis=0, how="any")
        if rets.empty:
            raise ValueError("No hay filas sin NaN en los retornos.")

        # Choose which return type to use in the optimization.
        # ret_kind = "log"    -> use log(1 + simple_return)
        # ret_kind = "simple" -> use simple_return as-is
        if ret_kind == "log":
            rets_used = np.log1p(rets)
        elif ret_kind in ("simple", "normal", "pct"):
            rets_used = rets
        else:
            raise ValueError("ret_kind debe ser 'log' o 'simple'.")

        tickers = list(rets_used.columns)
        n = len(tickers)

        rf = float(kwargs.get("rf_per_period", self.rf_per_period))

        mean_returns = rets_used.mean()
        cov_matrix = rets_used.cov()

        # Bounds.
        if bounds is not None:
            bounds_tuple = tuple(bounds)
            if len(bounds_tuple) != n:
                raise ValueError("bounds debe tener longitud = numero de activos.")
        else:
            if allow_short:
                bounds_tuple = tuple((-1.0, 1.0) for _ in range(n))
            else:
                bounds_tuple = tuple((0.0, 1.0) for _ in range(n))

        # Weight-sum constraint = 1.
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
            "objective": -float(result.fun),  # Maximum Sharpe reached
            "allow_short": allow_short,
            "ret_kind_used": ret_kind,
        }

        return w_opt, meta
