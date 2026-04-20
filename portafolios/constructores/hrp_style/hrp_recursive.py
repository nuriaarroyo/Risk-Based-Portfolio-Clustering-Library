from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .clustering.simple_cluster import hierarchical_clusters
from .distancias import corr, deprado


# Reuse the same distance registry.
DISTANCE_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "corr": corr.corr_distance,
    "deprado": deprado.de_prado_embedding_distance,
}


class HRPRecursive:
    """
    Lopez de Prado-style HRP with recursive bipartition (n_clusters = 2).

    - The asset set is always split into 2 sub-clusters.
    - The recursive function calls itself until the cluster has a single asset.
    - At each split, weights across the two sub-clusters are assigned
      through Naive Risk Parity (1 / sub-portfolio variance).

    It is compatible with `Portfolio.construir`:

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

        # Resolve distance function.
        if isinstance(distance, str):
            try:
                self.distance_func = DISTANCE_REGISTRY[distance]
                self.distance_name = distance
            except KeyError:
                raise ValueError(
                    f"Distancia desconocida: {distance}. "
                    f"Opciones validas: {list(DISTANCE_REGISTRY.keys())}"
                )
        elif callable(distance):
            self.distance_func = distance
            self.distance_name = getattr(distance, "__name__", "custom_distance")
        else:
            raise TypeError("distance debe ser string o funcion.")

        # Resolve clustering function.
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
            raise TypeError("clustering debe ser string o funcion.")

    # Public API.
    def optimizar(
        self,
        asset_returns: pd.DataFrame,
        **kwargs: Any,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Receive asset returns (dates x assets) and return:
            w: final HRP weights (Series, sum = 1)
            meta: metadata dictionary
        """
        if asset_returns is None or asset_returns.empty:
            raise ValueError("asset_returns no puede ser None ni vacio.")

        # Keep the last asset-return matrix in case plots need it later.
        self.last_asset_returns = asset_returns.copy()

        assets = list(asset_returns.columns)

        # Run the recursive routine on the full universe.
        w = self._recursive_bipartition(asset_returns, assets)

        # Normalize the result.
        w = w / w.sum()
        w = w.reindex(asset_returns.columns).fillna(0.0)

        # Basic metadata.
        meta: Dict[str, Any] = {
            "hrp_mode": "recursive_bipartition",
            "hrp_distance": self.distance_name,
            "hrp_clustering": self.clustering_name,
        }

        return w, meta

    @property
    def nombre(self) -> str:
        return self.display_name

    # Recursive core.
    def _recursive_bipartition(
        self,
        asset_returns: pd.DataFrame,
        assets: List[str],
    ) -> pd.Series:
        """
        Main recursive function.

        - If the cluster has a single asset -> weight 1.
        - Otherwise:
            1) split it into 2 sub-clusters with hierarchical clustering
            2) recurse on each side
            3) assign weights across both sub-portfolios with inverse variance
        """
        n = len(assets)

        # Base case: a single asset.
        if n == 1:
            return pd.Series({assets[0]: 1.0}, dtype=float)

        # Return submatrix for the current asset set only.
        sub_ret = asset_returns[assets]

        # 1) Distances within this cluster.
        dist = self.distance_func(sub_ret)

        # 2) Cluster into exactly 2 sub-clusters.
        clusters = self.clustering_func(dist, n_clusters=2)
        if len(clusters) != 2:
            # Guard against an unexpected clustering result.
            raise RuntimeError(f"Se esperaban 2 clusters, se obtuvieron {len(clusters)}")

        left_assets, right_assets = clusters[0], clusters[1]

        # 3) Recurse down both sides.
        w_left = self._recursive_bipartition(asset_returns, left_assets)
        w_right = self._recursive_bipartition(asset_returns, right_assets)

        # 4) Build the return series for each sub-portfolio.
        ret_left = (asset_returns[left_assets] @ w_left[left_assets]).rename("L")
        ret_right = (asset_returns[right_assets] @ w_right[right_assets]).rename("R")

        # 5) Variances for each sub-portfolio.
        var_left = float(ret_left.var())
        var_right = float(ret_right.var())

        # Avoid numerical issues.
        var_left = max(var_left, self.min_var)
        var_right = max(var_right, self.min_var)

        # 6) Naive Risk Parity allocation across the two sub-portfolios:
        #    w_cluster proportional to 1 / var_cluster
        inv_var_left = 1.0 / var_left
        inv_var_right = 1.0 / var_right
        total = inv_var_left + inv_var_right

        left_weight = inv_var_left / total
        right_weight = inv_var_right / total

        # 7) Scale internal weights by the cluster weight.
        w_left_scaled = w_left * left_weight
        w_right_scaled = w_right * right_weight

        # 8) Combine both sides.
        w = pd.concat([w_left_scaled, w_right_scaled])

        return w
