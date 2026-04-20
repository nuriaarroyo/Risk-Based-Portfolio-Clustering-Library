from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseConstructor


class NaiveRiskParity(BaseConstructor):
    """
    Naive Risk Parity (NRP) constructor.

    Assign weights proportional to inverse volatility:
        w_i proportional to 1 / sigma_i

    where sigma_i is the standard deviation of asset i returns.

    Typical usage:

        from portafolios.constructores.naive_risk_parity import NaiveRiskParity

        nrp = NaiveRiskParity()
        w, meta = nrp.optimizar(asset_returns)

    It is compatible with `PortfolioUniverse.construir`:

        p.construir(NaiveRiskParity())

    It can also be used as the inner/outer allocator in `HRPStyle`:

        hrp = HRPStyle(
            distance="deprado",
            clustering="hierarchical",
            inner=NaiveRiskParity(),
            outer=NaiveRiskParity(),
            n_clusters=3,
        )
    """

    def __init__(
        self,
        *,
        min_vol: float = 1e-8,
        display_name: str = "Naive Risk Parity (1/sigma)",
        nombre: str | None = None,
    ) -> None:
        """
        min_vol: volatility floor to avoid divide-by-zero issues.
        display_name: friendly label for tables, plots, and reports.
        """
        self.min_vol = float(min_vol)
        self.method_id = "naive_risk_parity"
        self.display_name = nombre or display_name

    def optimizar(
        self,
        asset_returns: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Receives:
            asset_returns: DataFrame (dates x assets)

        Returns:
            w: Series of weights (index = asset names, sum = 1)
            meta: dictionary with procedure metadata
        """
        if asset_returns is None or asset_returns.empty:
            raise ValueError("asset_returns no puede ser None ni vacio.")

        # 1) Per-asset volatilities.
        #    You can change ddof=1 or use .std() directly if preferred.
        sigma = asset_returns.std(ddof=1)

        # 2) Volatility floor to avoid 1/0.
        sigma_clipped = sigma.clip(lower=self.min_vol)

        # 3) Weights proportional to 1/sigma.
        inv_sigma = 1.0 / sigma_clipped
        w = inv_sigma / inv_sigma.sum()

        w = w.astype(float)
        w.name = "weights"

        meta: dict[str, Any] = {
            "nrp_method": "inverse_vol",
            "nrp_min_vol": self.min_vol,
            "nrp_sigma": sigma.to_dict(),
        }

        return w, meta
