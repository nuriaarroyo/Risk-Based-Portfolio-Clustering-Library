# portafolios/constructores/hrp_style.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Sequence
import pandas as pd

DistanceFunc = Callable[[pd.DataFrame], pd.DataFrame]
ClusteringFunc = Callable[[pd.DataFrame], list[list[str]]]
InnerConstructor = Callable[[pd.DataFrame], pd.Series]   # “constructor normal”
ClusterAllocator = Callable[[pd.DataFrame, list[list[str]]], pd.Series] 
# (pesos por cluster)


@dataclass
class HRPStyleConstructor:
    distance_func: DistanceFunc
    clustering_func: ClusteringFunc
    inner_constructor: InnerConstructor          # cómo optimizo dentro de cada cluster
    cluster_allocator: ClusterAllocator          # cómo reparto entre clusters (puede ser naive RP)

    def __call__(self, returns: pd.DataFrame) -> pd.Series:
        # 1) Distancia
        dist_matrix = self.distance_func(returns)

        # 2) Clustering -> lista de clusters, cada cluster es lista de tickers
        clusters = self.clustering_func(dist_matrix)

        # 3) Para cada cluster, construyo un sub-portafolio con tus constructores existentes
        local_weights_list: list[pd.Series] = []
        for assets in clusters:
            sub_ret = returns[assets]
            w_local = self.inner_constructor(sub_ret)  # usa Markowitz, min_var, lo que quieras
            # normalizo por si acaso
            w_local = w_local / w_local.sum()
            local_weights_list.append(w_local)

        # 4) Pesos por cluster (nivel “portafolio de portafolios”)
        cluster_weights = self.cluster_allocator(returns, clusters)
        # cluster_weights: index = identificador de cluster (0,1,2,...)

        # 5) Combinar: peso_total = peso_cluster * peso_local
        final_weights = pd.Series(0.0, index=returns.columns)

        for idx_cluster, assets in enumerate(clusters):
            w_cluster = cluster_weights.iloc[idx_cluster]
            w_local = local_weights_list[idx_cluster]
            final_weights[w_local.index] += w_cluster * w_local

        # sanity check
        final_weights = final_weights / final_weights.sum()
        return final_weights
