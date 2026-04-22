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

    @staticmethod
    def _make_feasible_initial_weights(bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)

        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("bounds must be finite for all assets.")
        if np.any(lower > upper):
            raise ValueError("Each lower bound must be <= the corresponding upper bound.")

        lower_sum = float(lower.sum())
        upper_sum = float(upper.sum())
        tol = 1e-10
        if lower_sum > 1.0 + tol or upper_sum < 1.0 - tol:
            raise ValueError("bounds are infeasible under the sum-to-one constraint.")

        x0 = lower.copy()
        remaining = 1.0 - lower_sum
        capacity = upper - lower
        total_capacity = float(capacity.sum())

        if remaining > tol:
            if total_capacity < remaining - tol:
                raise ValueError("bounds do not leave enough slack to satisfy the sum-to-one constraint.")
            if total_capacity <= tol:
                raise ValueError("bounds fully determine the weights and do not admit a feasible optimizer start.")
            x0 += remaining * (capacity / total_capacity)

        if not np.isclose(float(x0.sum()), 1.0, atol=1e-10):
            raise RuntimeError("Failed to build a feasible initial weight vector for the optimizer.")

        return x0

    @staticmethod
    def _validate_solution(
        weights: pd.Series,
        bounds: tuple[tuple[float, float], ...],
        *,
        allow_short: bool,
        tol: float = 1e-6,
    ) -> pd.Series:
        w = weights.astype(float).copy()
        if not np.isfinite(w.to_numpy()).all():
            raise RuntimeError("Optimizer returned non-finite weights.")

        if not allow_short:
            w[(w < 0) & (w > -tol)] = 0.0

        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        values = w.to_numpy(dtype=float)

        if np.any(values < lower - tol) or np.any(values > upper + tol):
            raise RuntimeError("Optimizer returned weights outside the requested bounds.")

        weight_sum = float(values.sum())
        if not np.isclose(weight_sum, 1.0, atol=tol):
            raise RuntimeError(f"Optimizer returned weights with sum {weight_sum:.8f} instead of 1.")

        if np.isclose(weight_sum, 0.0, atol=tol):
            raise RuntimeError("Optimizer returned a zero-sum weight vector.")

        if not np.isclose(weight_sum, 1.0, atol=1e-10):
            w = w / weight_sum

        return w

    @staticmethod
    def _handle_failure(
        *,
        on_failure: str,
        tickers: list[str],
        x0: np.ndarray,
        meta: dict[str, Any],
    ) -> tuple[pd.Series, dict[str, Any]]:
        if on_failure == "raise":
            raise RuntimeError(meta["message"])
        if on_failure == "use_initial_weights":
            fallback = pd.Series(x0, index=tickers, dtype=float)
            failure_meta = dict(meta)
            failure_meta.update({"fallback": "initial_weights"})
            return fallback, failure_meta
        raise ValueError("on_failure must be 'raise' or 'use_initial_weights'.")

    def optimizar(
        self,
        returns: pd.DataFrame,
        *,
        ret_kind: str = "simple",
        allow_short: bool = False,
        bounds: Any = None,
        on_failure: str = "raise",
        **kwargs,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if returns is None or returns.empty:
            raise ValueError("The returns DataFrame is empty or None.")

        # Returns coming from PortfolioUniverse are simple returns.
        rets = returns.dropna(axis=0, how="any")
        if rets.empty:
            raise ValueError("There are no rows without NaN values in the returns data.")

        # Choose which return type to use in the optimization.
        # ret_kind = "log"    -> use log(1 + simple_return)
        # ret_kind = "simple" -> use simple_return as-is
        if ret_kind == "log":
            rets_used = np.log1p(rets)
        elif ret_kind in ("simple", "normal", "pct"):
            rets_used = rets
        else:
            raise ValueError("ret_kind must be 'log' or 'simple'.")

        tickers = list(rets_used.columns)
        n = len(tickers)

        rf = float(kwargs.get("rf_per_period", self.rf_per_period))

        mean_returns = rets_used.mean()
        cov_matrix = rets_used.cov()

        # Bounds.
        if bounds is not None:
            bounds_tuple = tuple(bounds)
            if len(bounds_tuple) != n:
                raise ValueError("bounds must have the same length as the number of assets.")
        else:
            if allow_short:
                bounds_tuple = tuple((-1.0, 1.0) for _ in range(n))
            else:
                bounds_tuple = tuple((0.0, 1.0) for _ in range(n))

        # Weight-sum constraint = 1.
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        x0 = self._make_feasible_initial_weights(bounds_tuple)

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
            meta = {
                "n_assets": n,
                "success": False,
                "message": f"Markowitz optimization failed: {result.message}",
                "rf_per_period": rf,
                "objective": None,
                "allow_short": allow_short,
                "ret_kind_used": ret_kind,
                "on_failure": on_failure,
            }
            return self._handle_failure(
                on_failure=on_failure,
                tickers=tickers,
                x0=x0,
                meta=meta,
            )

        w_opt = self._validate_solution(
            pd.Series(result.x, index=tickers, dtype=float),
            bounds_tuple,
            allow_short=allow_short,
        )

        meta = {
            "n_assets": n,
            "success": True,
            "message": result.message,
            "rf_per_period": rf,
            "objective": -float(result.fun),  # Maximum Sharpe reached
            "allow_short": allow_short,
            "ret_kind_used": ret_kind,
            "on_failure": on_failure,
        }

        return w_opt, meta
