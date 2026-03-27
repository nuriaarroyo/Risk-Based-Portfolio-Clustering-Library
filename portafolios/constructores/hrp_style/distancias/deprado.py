# portafolios/constructores/hrp_style/distancias/deprado.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def de_prado_corr_distance(returns: pd.DataFrame) -> pd.DataFrame:
    r"""
    Primer paso de López de Prado:
        d_ij = sqrt( 0.5 * (1 - corr_ij) )

    returns: DataFrame (fechas x activos)

    return: matriz de distancias D (assets x assets)
    """
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1.0 - corr))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def de_prado_embedding_distance(returns: pd.DataFrame) -> pd.DataFrame:
    r"""
    Distancia "completa" del ejemplo del paper:

    1) Construye D con d_ij = sqrt(0.5 * (1 - corr_ij))
    2) Define la distancia entre activos como la distancia euclidiana
       entre los vectores-columna de D:

           \hat d_ij = || D_i - D_j ||_2

    returns: DataFrame (fechas x activos)

    return: matriz de distancias \hat D (assets x assets)
    """
    # Paso 1: matriz D con la distancia de correlación
    D = de_prado_corr_distance(returns)

    # Paso 2: distancia euclidiana entre columnas (o filas, es simétrica)
    # Usamos scipy.pdist sobre las filas de D; como D es simétrica, da igual
    dist_condensed = pdist(D.values, metric="euclidean")
    dist_matrix = squareform(dist_condensed)

    hatD = pd.DataFrame(dist_matrix, index=D.index, columns=D.index)
    return hatD
