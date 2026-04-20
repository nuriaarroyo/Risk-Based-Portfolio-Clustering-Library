from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def returns_simple(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna()


def returns_log(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t) - ln(P_{t-1})."""
    return np.log(prices).diff().dropna()


def mean_return(returns: pd.DataFrame, ann_factor: Optional[int] = None) -> pd.Series:
    """
    Per-period mean of the returns DataFrame.
    - If `ann_factor` is None -> per-period mean.
    - If `ann_factor` is an int -> annualize as mean * ann_factor.
    """
    mu = returns.mean()
    return mu if ann_factor is None else mu * ann_factor


def volatility(
    returns: pd.DataFrame,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> pd.Series:
    """
    Standard-deviation volatility of the returns DataFrame.
    - If `ann_factor` is None -> per-period standard deviation.
    - If `ann_factor` is an int -> annualize as std * sqrt(ann_factor).
    """
    sigma = returns.std(ddof=ddof)
    return sigma if ann_factor is None else sigma * np.sqrt(ann_factor)


def covariance_matrix(
    returns: pd.DataFrame,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Asset covariance matrix.
    - If `ann_factor` is None -> per-period covariance.
    - If `ann_factor` is an int -> annualize as cov * ann_factor.
    """
    cov = returns.cov(ddof=ddof)
    return cov if ann_factor is None else cov * ann_factor


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Asset correlation matrix.
    """
    return returns.corr()
