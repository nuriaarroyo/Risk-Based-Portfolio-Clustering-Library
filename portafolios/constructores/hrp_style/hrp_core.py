from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from portafolios.constructores.equal_weight import EqualWeightConstructor

from .clustering.simple_cluster import hierarchical_cluster_details, hierarchical_clusters
from .distancias import corr, deprado


DISTANCE_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "corr": corr.corr_distance,
    "deprado": deprado.de_prado_embedding_distance,
}


class HRPStyle:
    """
    Two-level HRP-style constructor compatible with `PortfolioUniverse.construir`.

    Typical usage:

        from portafolios.constructores.equal_weight import EqualWeightConstructor
        from portafolios.constructores.hrp_style.hrp_core import HRPStyle

        hrp = HRPStyle(
            distance="corr",
            clustering="hierarchical",
            inner=EqualWeightConstructor(),   # constructor inside each cluster
            outer=EqualWeightConstructor(),   # constructor across clusters
            n_clusters=3,
        )

        p.construir(hrp)
    """

    def __init__(
        self,
        *,
        distance: str | Callable[[pd.DataFrame], pd.DataFrame] = "corr",
        clustering: str | Callable[[pd.DataFrame, int], list[list[str]]] = "hierarchical",
        inner: Any | None = None,
        outer: Any | None = None,
        n_clusters: int = 3,
        display_name: str = "HRP Style",
        nombre: str | None = None,
    ) -> None:
        self.method_id = "hrp_style"
        self.display_name = nombre or display_name
        self.n_clusters = n_clusters

        # Resolve distance function.
        if isinstance(distance, str):
            try:
                self.distance_func = DISTANCE_REGISTRY[distance]
                self.distance_name = distance
            except KeyError:
                raise ValueError(
                    f"Unknown distance: {distance}. "
                    f"Valid options: {list(DISTANCE_REGISTRY.keys())}"
                )
        elif callable(distance):
            self.distance_func = distance
            self.distance_name = getattr(distance, "__name__", "custom_distance")
        else:
            raise TypeError("distance must be a string from DISTANCE_REGISTRY or a callable.")

        # Resolve clustering function.
        if isinstance(clustering, str):
            if clustering == "hierarchical":
                # Our simple hierarchical helper accepts (dist, n_clusters=...).
                self.clustering_func = hierarchical_clusters
                self.clustering_name = "hierarchical"
            else:
                raise ValueError(f"Unknown clustering method: {clustering}")
        elif callable(clustering):
            self.clustering_func = clustering
            self.clustering_name = getattr(clustering, "__name__", "custom_clustering")
        else:
            raise TypeError("clustering must be 'hierarchical' or a callable.")

        # Resolve the inner and outer constructors.
        # If nothing is provided, use EqualWeightConstructor() at both levels.
        self.inner_constructor = inner if inner is not None else EqualWeightConstructor()
        self.outer_constructor = outer if outer is not None else EqualWeightConstructor()

    # Helper to run any constructor (Markowitz, Naive, etc.).
    def _run_constructor(
        self,
        constructor: Any,
        asset_returns: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.Series, dict]:
        """
        Accepts:
          - objects with `.optimizar(returns, **kwargs)` returning `(w, meta)` or `w`
          - callables with `(returns, **kwargs)` returning `(w, meta)` or `w`
        """
        if hasattr(constructor, "optimizar"):
            res = constructor.optimizar(asset_returns, **kwargs)
        else:
            res = constructor(asset_returns, **kwargs)

        if isinstance(res, tuple):
            w, meta = res
        else:
            w, meta = res, {}

        meta = meta or {}

        # Make sure we always end up with a properly indexed Series.
        if not isinstance(w, pd.Series):
            w = pd.Series(w, index=asset_returns.columns, dtype=float)

        return w, meta

    @property
    def nombre(self) -> str:
        return self.display_name

    def optimizar(
        self,
        asset_returns: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.Series, dict]:
        """
        Method used by `Portfolio.construir`.
        It receives asset returns (dates x assets) and returns `(weights, meta)`.
        """
        n_clusters = kwargs.pop("n_clusters", self.n_clusters)

        # 1) Distances.
        dist = self.distance_func(asset_returns)

        # 2) Clusters.
        clustering_details = None
        if self.clustering_func is hierarchical_clusters:
            clustering_details = hierarchical_cluster_details(dist, n_clusters=n_clusters)
            clusters = clustering_details.clusters
        else:
            clusters = self.clustering_func(dist, n_clusters=n_clusters)

        cluster_returns: dict[str, pd.Series] = {}
        local_weights: dict[str, pd.Series] = {}
        cluster_assets: dict[str, list[str]] = {}
        inner_meta_by_cluster: dict[str, dict[str, Any]] = {}

        meta: dict[str, Any] = {
            "hrp_n_clusters": n_clusters,
            "hrp_distance": self.distance_name,
            "hrp_clustering": self.clustering_name,
            "hrp_clusters": clusters,
        }
        if clustering_details is not None:
            meta["hrp_cluster_labels"] = clustering_details.labels.to_dict()
            meta["hrp_linkage_matrix"] = clustering_details.linkage_matrix.tolist()

        # 3) Build each cluster locally.
        for idx, assets in enumerate(clusters):
            name = f"C{idx}"
            sub_ret = asset_returns[assets]

            w_local, inner_meta = self._run_constructor(self.inner_constructor, sub_ret)
            inner_meta_by_cluster[name] = dict(inner_meta or {})

            w_local = w_local / w_local.sum()
            local_weights[name] = w_local
            cluster_assets[name] = list(assets)
            # Build the cluster-level return series that will feed the outer allocator.
            cluster_returns[name] = (sub_ret @ w_local).rename(name)

        # 4) Portfolio of portfolios (clusters treated as "assets").
        clusters_df = pd.DataFrame(cluster_returns)  # dates x clusters
        w_clusters, outer_meta = self._run_constructor(self.outer_constructor, clusters_df)

        w_clusters = w_clusters / w_clusters.sum()

        # 5) Combine: W_cluster * w_local.
        final_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)

        for name, w_local in local_weights.items():
            cluster_weight = float(w_clusters[name])
            final_weights[w_local.index] += cluster_weight * w_local

        final_weights = final_weights / final_weights.sum()

        # Final metadata. Adjust this block if you want to store more diagnostics.
        shared_inner_meta: dict[str, Any] = {}
        if inner_meta_by_cluster:
            shared_keys = set.intersection(*(set(cluster_meta.keys()) for cluster_meta in inner_meta_by_cluster.values()))
            for key in shared_keys:
                values = [cluster_meta[key] for cluster_meta in inner_meta_by_cluster.values()]
                first_value = values[0]
                if all(value == first_value for value in values[1:]):
                    shared_inner_meta[key] = first_value
        meta["hrp_cluster_assets"] = cluster_assets
        meta["hrp_inner_meta_by_cluster"] = inner_meta_by_cluster
        meta["hrp_outer_meta"] = dict(outer_meta or {})
        meta.update({f"hrp_inner_{k}": v for k, v in shared_inner_meta.items()})
        meta.update({f"hrp_outer_{k}": v for k, v in (outer_meta or {}).items()})

        # Keep diagnostics on the HRPStyle instance itself.
        self.last_dist = dist
        self.last_clusters = clusters
        self.last_cluster_returns = pd.DataFrame(cluster_returns)
        self.last_w_clusters = w_clusters
        self.last_local_weights = local_weights
        self.last_final_weights = final_weights
        self.last_cluster_assets = cluster_assets
        self.last_inner_meta_by_cluster = inner_meta_by_cluster
        self.last_outer_meta = dict(outer_meta or {})
        self.last_linkage_matrix = (
            clustering_details.linkage_matrix.copy() if clustering_details is not None else None
        )
        self.last_cluster_labels = (
            clustering_details.labels.copy() if clustering_details is not None else None
        )

        return final_weights, meta

    def __call__(self, asset_returns: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Convenience path if you want to call HRPStyle directly: `hrp(returns)`.
        `Portfolio.construir` does not use this method; it calls `.optimizar`.
        """
        w, _ = self.optimizar(asset_returns, **kwargs)
        return w
