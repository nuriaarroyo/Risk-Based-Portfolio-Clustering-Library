from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import pandas as pd
import numpy as np

from .distancias import corr, deprado
from .clustering.simple_cluster import hierarchical_clusters


# Puedes reutilizar el mismo registry de distancias
DISTANCE_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "corr": corr.corr_distance,
    "deprado": deprado.de_prado_embedding_distance,
}


class HRPRecursive:
    """
    HRP estilo López de Prado con bipartición recursiva (n_clusters = 2).

    - Siempre se divide el conjunto de activos en 2 sub-clusters.
    - La función recursiva se llama a sí misma hasta que el cluster tiene 1 solo activo.
    - En cada bipartición, los pesos entre los 2 sub-clusters se asignan
      por Naive Risk Parity (1/varianza del sub-portafolio).

    Es compatible con Portfolio.construir:

        from portafolios.constructores.hrp_style.hrp_recursive import HRPRecursive

        hrp_rec = HRPRecursive(distance="deprado", clustering="hierarchical")
        p.construir(hrp_rec)
    """

    def __init__(
        self,
        *,
        distance: str | Callable[[pd.DataFrame], pd.DataFrame] = "corr",
        clustering: str | Callable[[pd.DataFrame, int], List[List[str]]] = "hierarchical",
        display_name: str = "HRP Recursive",
        nombre: str | None = None,
        min_var: float = 1e-8,
    ) -> None:
        self.method_id = "hrp_recursive"
        self.display_name = nombre or display_name
        self.min_var = float(min_var)

        # ----- resolver distancia -----
        if isinstance(distance, str):
            try:
                self.distance_func = DISTANCE_REGISTRY[distance]
                self.distance_name = distance
            except KeyError:
                raise ValueError(
                    f"Distancia desconocida: {distance}. "
                    f"Opciones válidas: {list(DISTANCE_REGISTRY.keys())}"
                )
        elif callable(distance):
            self.distance_func = distance
            self.distance_name = getattr(distance, "__name__", "custom_distance")
        else:
            raise TypeError("distance debe ser string o función.")

        # ----- resolver clustering -----
        if isinstance(clustering, str):
            if clustering == "hierarchical":
                self.clustering_func = hierarchical_clusters
                self.clustering_name = "hierarchical"
            else:
                raise ValueError(f"Clustering desconocido: {clustering}")
        elif callable(clustering):
            self.clustering_func = clustering
            self.clustering_name = getattr(clustering, "__name__", "custom_clustering")
        else:
            raise TypeError("clustering debe ser string o función.")

    # -------------------- API pública --------------------

    def optimizar(
        self,
        asset_returns: pd.DataFrame,
        **kwargs: Any,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Recibe retornos de assets (fechas x activos) y regresa:
            w: pesos finales HRP (Serie, suma = 1)
            meta: diccionario con metadatos
        """
        if asset_returns is None or asset_returns.empty:
            raise ValueError("asset_returns no puede ser None ni vacío.")

        # Guardamos por si luego quieres plots
        self.last_asset_returns = asset_returns.copy()

        assets = list(asset_returns.columns)

        # Llamamos a la función recursiva sobre TODO el universo
        w = self._recursive_bipartition(asset_returns, assets)

        # Aseguramos normalización
        w = w / w.sum()
        w = w.reindex(asset_returns.columns).fillna(0.0)

        # Meta básica
        meta: Dict[str, Any] = {
            "hrp_mode": "recursive_bipartition",
            "hrp_distance": self.distance_name,
            "hrp_clustering": self.clustering_name,
        }

        return w, meta

    @property
    def nombre(self) -> str:
        return self.display_name

    # -------------------- núcleo recursivo --------------------

    def _recursive_bipartition(
        self,
        asset_returns: pd.DataFrame,
        assets: List[str],
    ) -> pd.Series:
        """
        Función recursiva principal.

        - Si el cluster tiene 1 solo activo -> peso 1.
        - Si tiene más:
            1) lo partimos en 2 sub-clusters usando clustering jerárquico
            2) recursión en cada lado
            3) asignamos pesos entre los dos sub-portafolios por inverse-variance
        """
        n = len(assets)

        # Caso base: un solo activo
        if n == 1:
            return pd.Series({assets[0]: 1.0}, dtype=float)

        # Sub-matriz de retornos solo para estos assets
        sub_ret = asset_returns[assets]

        # 1) Distancias dentro de este cluster
        dist = self.distance_func(sub_ret)

        # 2) Clustering en EXACTAMENTE 2 sub-clusters
        clusters = self.clustering_func(dist, n_clusters=2)
        if len(clusters) != 2:
            # por si acaso el algoritmo devuelve algo raro
            raise RuntimeError(f"Se esperaban 2 clusters, se obtuvieron {len(clusters)}")

        left_assets, right_assets = clusters[0], clusters[1]

        # 3) Recursión en cada lado
        w_left = self._recursive_bipartition(asset_returns, left_assets)
        w_right = self._recursive_bipartition(asset_returns, right_assets)

        # 4) Construimos las series de retornos de cada sub-portafolio
        ret_left = (asset_returns[left_assets] @ w_left[left_assets]).rename("L")
        ret_right = (asset_returns[right_assets] @ w_right[right_assets]).rename("R")

        # 5) Varianzas de cada sub-portafolio
        var_left = float(ret_left.var())
        var_right = float(ret_right.var())

        # Evitar problemas numéricos
        var_left = max(var_left, self.min_var)
        var_right = max(var_right, self.min_var)

        # 6) Asignación Naive Risk Parity entre los DOS sub-portafolios:
        #    w_cluster ∝ 1 / var_cluster
        inv_var_left = 1.0 / var_left
        inv_var_right = 1.0 / var_right
        total = inv_var_left + inv_var_right

        W_left = inv_var_left / total
        W_right = inv_var_right / total

        # 7) Escalamos pesos internos por el peso del cluster
        w_left_scaled = w_left * W_left
        w_right_scaled = w_right * W_right

        # 8) Combinamos
        w = pd.concat([w_left_scaled, w_right_scaled])

        return w
