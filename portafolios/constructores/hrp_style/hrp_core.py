# portafolios/constructores/hrp_style/hrp_core.py
from __future__ import annotations

from typing import Callable, Any
import pandas as pd

from .distancias  import corr, deprado
from .clustering.simple_cluster import hierarchical_clusters


from portafolios.constructores.equal_weight import EqualWeightConstructor

DISTANCE_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "corr": corr.corr_distance,
        "deprado": deprado.de_prado_embedding_distance,  # nombre como tú quieras  
    }


class HRPStyle:
    """
    Constructor HRP-style de dos niveles, compatible con PortfolioUniverse.construir.

    Se usa así:

        from portafolios.constructores.hrp_style.hrp_core import HRPStyle
        from portafolios.constructores.equal_weight import EqualWeightConstructor

        hrp = HRPStyle(
            distance="corr",
            clustering="hierarchical",
            inner=EqualWeightConstructor(),   # constructor dentro de cada cluster
            outer=EqualWeightConstructor(),   # constructor entre clusters
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

        
        # ----- resolver distancia -----
        if isinstance(distance, str):
                try:
                    self.distance_func = DISTANCE_REGISTRY[distance]
                except KeyError:
                    raise ValueError(
                        f"Distancia desconocida: {distance}. "
                        f"Opciones válidas: {list(DISTANCE_REGISTRY.keys())}"
                    )
        elif callable(distance):
                self.distance_func = distance
        else:
                raise TypeError("distance debe ser un string de DISTANCE_REGISTRY o una función.")


        # ----- resolver clustering -----
        if isinstance(clustering, str):
            if clustering == "hierarchical":
                # nuestra función simple_hierarchical(dist, n_clusters=...)
                self.clustering_func = hierarchical_clusters
            else:
                raise ValueError(f"Clustering desconocido: {clustering}")
        elif callable(clustering):
            self.clustering_func = clustering
        else:
            raise TypeError("clustering debe ser 'hierarchical' o una función.")

        # ----- constructores internos -----
        # Si no te pasan nada, usa EqualWeightConstructor() en ambos niveles
        self.inner_constructor = inner if inner is not None else EqualWeightConstructor()
        self.outer_constructor = outer if outer is not None else EqualWeightConstructor()

    # helper para correr cualquier constructor (Markowitz, Naive, etc.)
    def _run_constructor(
        self,
        constructor: Any,
        asset_returns: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.Series, dict]:
        """
        Acepta:
          - objetos con .optimizar(returns, **kwargs) -> (w, meta) o w
          - o funciones callables(returns, **kwargs) -> (w, meta) o w
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

        # Nos aseguramos de tener una Serie bien indexada
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
        Método que usa Portfolio.construir.
        Recibe retornos de assets (fechas x activos) y regresa (weights, meta).
        """
        n_clusters = kwargs.pop("n_clusters", self.n_clusters)

        # 1) Distancias
        dist = self.distance_func(asset_returns)

        # 2) Clusters
        clusters = self.clustering_func(dist, n_clusters=n_clusters)

        cluster_returns: dict[str, pd.Series] = {}
        local_weights: dict[str, pd.Series] = {}

        meta: dict[str, Any] = {
            "hrp_n_clusters": n_clusters,
            "hrp_distance": getattr(self.distance_func, "__name__", str(self.distance_func)),
            "hrp_clustering": getattr(self.clustering_func, "__name__", str(self.clustering_func)),
            "hrp_clusters": clusters,
        }

        # 3) Dentro de cada cluster
        last_inner_meta: dict[str, Any] = {}
        for idx, assets in enumerate(clusters):
            name = f"C{idx}"
            sub_ret = asset_returns[assets]

            w_local, inner_meta = self._run_constructor(self.inner_constructor, sub_ret)
            last_inner_meta = inner_meta or last_inner_meta

            w_local = w_local / w_local.sum()
            local_weights[name] = w_local
            cluster_returns[name] = (sub_ret @ w_local).rename(name) #ACA SE CONTRUYE LA SERE DE PORTAFOIOLIOS DE CADA CLUSTER

        # 4) Portafolio de portafolios (clusters como "activos")
        clusters_df = pd.DataFrame(cluster_returns)  # fechas x clusters
        w_clusters, outer_meta = self._run_constructor(self.outer_constructor, clusters_df)

        w_clusters = w_clusters / w_clusters.sum()

        # 5) Combinar: W_cluster * w_local
        final_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)

        for name, w_local in local_weights.items():
            W_c = float(w_clusters[name])
            final_weights[w_local.index] += W_c * w_local

        final_weights = final_weights / final_weights.sum()

        # meta final (puedes ajustar lo que quieras guardar)
        meta.update({f"hrp_inner_{k}": v for k, v in (last_inner_meta or {}).items()})
        meta.update({f"hrp_outer_{k}": v for k, v in (outer_meta or {}).items()})

        
        # Guardar diagnósticos en el propio HRPStyle
        self.last_dist = dist
        self.last_clusters = clusters
        self.last_cluster_returns = pd.DataFrame(cluster_returns)
        self.last_w_clusters = w_clusters
        self.last_local_weights = local_weights
        self.last_final_weights = final_weights

        return final_weights, meta


    def __call__(self, asset_returns: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Por si quieres usar HRPStyle como función directa: hrp(returns).
        Portfolio.construir NO usa esto, sino .optimizar.
        """
        w, _ = self.optimizar(asset_returns, **kwargs)
        return w


