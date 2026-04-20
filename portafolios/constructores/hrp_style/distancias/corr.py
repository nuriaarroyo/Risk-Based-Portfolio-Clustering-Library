from __future__ import annotations

import numpy as np
import pandas as pd


def corr_distance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Simple correlation-based distance:

        D_ij = 1 - corr_ij

    returns: DataFrame (dates x assets)
    """
    corr = returns.corr()
    dist = 1.0 - corr
    dist = pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    return dist
