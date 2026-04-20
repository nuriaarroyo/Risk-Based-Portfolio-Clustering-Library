from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def de_prado_corr_distance(returns: pd.DataFrame) -> pd.DataFrame:
    r"""
    First Lopez de Prado step:
        d_ij = sqrt(0.5 * (1 - corr_ij))

    returns: DataFrame (dates x assets)

    return: distance matrix D (assets x assets)
    """
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1.0 - corr))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def de_prado_embedding_distance(returns: pd.DataFrame) -> pd.DataFrame:
    r"""
    "Full" distance from the paper example:

    1) Build D with d_ij = sqrt(0.5 * (1 - corr_ij))
    2) Define the distance between assets as the Euclidean distance
       between the column vectors of D:

           \hat d_ij = || D_i - D_j ||_2

    returns: DataFrame (dates x assets)

    return: distance matrix \hat D (assets x assets)
    """
    # Step 1: matrix D with correlation distance.
    dist_matrix = de_prado_corr_distance(returns)

    # Step 2: Euclidean distance between columns (or rows; D is symmetric).
    # We use scipy.pdist on the rows of D, which is equivalent here.
    dist_condensed = pdist(dist_matrix.values, metric="euclidean")
    embedded_matrix = squareform(dist_condensed)

    hat_d = pd.DataFrame(embedded_matrix, index=dist_matrix.index, columns=dist_matrix.index)
    return hat_d
