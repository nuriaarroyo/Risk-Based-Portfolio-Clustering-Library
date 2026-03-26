# portafolios/constructores/naive_rp.py
from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from .base import BaseConstructor


class NaiveRiskParity(BaseConstructor):
    """
    Constructor Naive Risk Parity (NRP).

    Asigna pesos proporcionales al inverso de la volatilidad:
        w_i ∝ 1 / sigma_i

    Donde sigma_i es la desviación estándar de los retornos del activo i.

    Uso típico:

        from portafolios.constructores.naive_risk_parity import NaiveRiskParity

        nrp = NaiveRiskParity()
        w, meta = nrp.optimizar(asset_returns)

    Es compatible con Portfolio.construir:

        p.construir(NaiveRiskParity())

    y también puede usarse como inner/outer en HRPStyle:

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
        nombre: str = "Naive Risk Parity (1/sigma)",
    ) -> None:
        """
        min_vol: piso para la volatilidad, para evitar divisiones por cero.
        nombre: etiqueta amigable para guardar en p.info["constructor"].
        """
        self.min_vol = float(min_vol)
        self.nombre = nombre

    def optimizar(
        self,
        asset_returns: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Recibe:
            asset_returns: DataFrame (fechas x activos)

        Regresa:
            w: Serie con pesos (index = nombres de activos, suma = 1)
            meta: diccionario con metadatos del procedimiento
        """
        if asset_returns is None or asset_returns.empty:
            raise ValueError("asset_returns no puede ser None ni vacío.")

        # 1) Volatilidades por activo
        #    (puedes cambiar ddof=1 ó usar .std() tal cual)
        sigma = asset_returns.std(ddof=1)

        # 2) Piso de volatilidad para evitar 1/0
        sigma_clipped = sigma.clip(lower=self.min_vol)

        # 3) Pesos proporcionales a 1/sigma
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
