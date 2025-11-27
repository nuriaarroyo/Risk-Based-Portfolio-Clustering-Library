# portafolios/constructores/hrp_style/distancias/corr.py
from __future__ import annotations

import pandas as pd
import numpy as np


def corr_distance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Distancia simple basada en correlación:

        D_ij = 1 - corr_ij

    returns: DataFrame (fechas x activos)
    """
    corr = returns.corr()
    dist = 1.0 - corr
    dist = pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    return dist
