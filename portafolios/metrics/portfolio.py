from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1) MOMENT-BASED KPIs (use mu, cov, weights)
#    - Recommended: mu = mean of simple returns (per period)
#                   cov = covariance matrix (simple or log, but stay consistent)
# ============================================================


def expected_return_from_moments(
    mu: pd.Series,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
) -> float:
    """
    Expected return: mu_p = w' mu (per period if ann_factor=None).
    """
    w = weights.reindex(mu.index).fillna(0).values
    er = float(np.dot(w, mu.values))
    return er if ann_factor is None else er * ann_factor


def expected_volatility_from_moments(
    cov: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
) -> float:
    """
    Expected volatility: sigma_p = sqrt(w' Sigma w).
    """
    cov_values = cov.reindex(index=weights.index, columns=weights.index).fillna(0).values
    w = weights.fillna(0).values
    var = float(w @ cov_values @ w)
    vol = np.sqrt(var) if var >= 0 else np.nan
    return vol if ann_factor is None else vol * np.sqrt(ann_factor)


def sharpe_from_moments(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    *,
    rf_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
) -> float:
    """
    Moment-based Sharpe: S = (mu_p - rf) / sigma_p.
    `rf_per_period` must be in the same units as `mu`.
    """
    er = expected_return_from_moments(mu, weights, ann_factor=None)
    vol = expected_volatility_from_moments(cov, weights, ann_factor=None)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    sharpe = (er - rf_per_period) / vol
    return sharpe if ann_factor is None else sharpe * np.sqrt(ann_factor)


def risk_contributions_from_cov(
    cov: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    as_fraction: bool = True,
) -> pd.Series:
    """
    Volatility risk contribution for each asset.

    Component contribution to portfolio volatility:
        RC_i = w_i * (Sigma w)_i / sigma_p

    Fractional contribution:
        FRC_i = RC_i / sigma_p

    Returns fractions when `as_fraction=True`, otherwise returns component
    contributions that add up to portfolio volatility.
    """
    aligned_weights = weights.fillna(0.0).astype(float)
    sigma_matrix = cov.reindex(index=aligned_weights.index, columns=aligned_weights.index).fillna(0.0).values
    if ann_factor is not None:
        sigma_matrix = sigma_matrix * ann_factor
    w = aligned_weights.values.reshape(-1, 1)
    portfolio_variance = float((w.T @ sigma_matrix @ w).item())
    port_vol = float(np.sqrt(portfolio_variance))
    if port_vol == 0 or np.isnan(port_vol):
        return pd.Series(np.nan, index=aligned_weights.index)
    marginal = (sigma_matrix @ w)[:, 0]
    component_rc = (aligned_weights.values * marginal) / port_vol
    if as_fraction:
        return pd.Series(component_rc / port_vol, index=aligned_weights.index)
    return pd.Series(component_rc, index=aligned_weights.index)


# ============================================================
# 2) PATH-BASED KPIs (require a portfolio return series)
#    - Useful for MDD, Sortino, VaR/CVaR, TE/IR, Alpha/Beta
# ============================================================


def _align_rw(returns: pd.DataFrame, weights: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    returns = returns.reindex(columns=weights.index)
    weights = weights.reindex(returns.columns).fillna(0.0)
    return returns, weights


def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Portfolio return series: r_p,t = R_t @ w (per period).
    """
    rets, w = _align_rw(returns, weights)
    return (rets @ w).dropna()


def cumulative_return_series(portfolio_returns: pd.Series) -> pd.Series:
    rp = portfolio_returns.dropna()
    return (1.0 + rp).cumprod() - 1.0


def drawdown_series(portfolio_returns: pd.Series) -> pd.Series:
    cumulative_returns = cumulative_return_series(portfolio_returns)
    wealth = 1.0 + cumulative_returns
    return wealth / wealth.cummax() - 1.0


def realized_total_return_from_series(portfolio_returns: pd.Series) -> float:
    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    return float((1.0 + rp).prod() - 1.0)


def realized_annualized_return_from_series(
    portfolio_returns: pd.Series,
    *,
    ann_factor: int,
) -> float:
    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    total_return = realized_total_return_from_series(rp)
    return float((1.0 + total_return) ** (ann_factor / len(rp)) - 1.0)


def realized_annualized_volatility_from_series(
    portfolio_returns: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    rp = portfolio_returns.dropna()
    if len(rp) <= 1:
        return 0.0
    sigma = float(rp.std(ddof=ddof))
    return sigma if ann_factor is None else float(sigma * np.sqrt(ann_factor))


def sharpe_from_series(
    portfolio_returns: pd.Series,
    *,
    rf_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    mu = float(rp.mean())
    sigma = float(rp.std(ddof=ddof)) if len(rp) > 1 else 0.0
    if sigma == 0.0 or np.isnan(sigma):
        return float("nan")
    sharpe_ratio = (mu - rf_per_period) / sigma
    return sharpe_ratio if ann_factor is None else float(sharpe_ratio * np.sqrt(ann_factor))


def max_drawdown_from_series(portfolio_returns: pd.Series) -> float:
    drawdown = drawdown_series(portfolio_returns)
    if drawdown.empty:
        return float("nan")
    return float(drawdown.min())


def realized_total_return(returns: pd.DataFrame, weights: pd.Series) -> float:
    return realized_total_return_from_series(portfolio_return_series(returns, weights))


def realized_annualized_return(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: int,
) -> float:
    return realized_annualized_return_from_series(
        portfolio_return_series(returns, weights),
        ann_factor=ann_factor,
    )


def realized_annualized_volatility(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return realized_annualized_volatility_from_series(
        portfolio_return_series(returns, weights),
        ann_factor=ann_factor,
        ddof=ddof,
    )


def max_drawdown(returns: pd.DataFrame, weights: pd.Series) -> float:
    return max_drawdown_from_series(portfolio_return_series(returns, weights))


def downside_deviation(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    mar_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
) -> float:
    return downside_deviation_from_series(
        portfolio_return_series(returns, weights),
        mar_per_period=mar_per_period,
        ann_factor=ann_factor,
    )


def downside_deviation_from_series(
    portfolio_returns: pd.Series,
    *,
    mar_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
) -> float:
    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    downside = np.clip(rp - mar_per_period, None, 0.0)
    dd = np.sqrt((downside**2).mean())
    return dd if ann_factor is None else dd * np.sqrt(ann_factor)


def sortino(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    mar_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
) -> float:
    return sortino_from_series(
        portfolio_return_series(returns, weights),
        mar_per_period=mar_per_period,
        ann_factor=ann_factor,
    )


def sortino_from_series(
    portfolio_returns: pd.Series,
    *,
    mar_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
) -> float:
    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    er = float(rp.mean())
    dd = downside_deviation_from_series(rp, mar_per_period=mar_per_period, ann_factor=None)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    sortino_ratio = (er - mar_per_period) / dd
    return sortino_ratio if ann_factor is None else sortino_ratio * np.sqrt(ann_factor)


# Gaussian VaR/CVaR from the realized path.
def var_gaussian(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    alpha: float = 0.95,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return var_gaussian_from_series(
        portfolio_return_series(returns, weights),
        alpha=alpha,
        ann_factor=ann_factor,
        ddof=ddof,
    )


def var_gaussian_from_series(
    portfolio_returns: pd.Series,
    *,
    alpha: float = 0.95,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    from scipy.stats import norm

    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    mu, sigma = rp.mean(), rp.std(ddof=ddof)
    if np.isnan(mu) or np.isnan(sigma):
        return float("nan")
    if ann_factor is not None:
        mu = mu * ann_factor
        sigma = sigma * np.sqrt(ann_factor)
    z = norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    return float(max(var, 0.0))


def cvar_gaussian(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    alpha: float = 0.95,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return cvar_gaussian_from_series(
        portfolio_return_series(returns, weights),
        alpha=alpha,
        ann_factor=ann_factor,
        ddof=ddof,
    )


def cvar_gaussian_from_series(
    portfolio_returns: pd.Series,
    *,
    alpha: float = 0.95,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    from scipy.stats import norm

    rp = portfolio_returns.dropna()
    if rp.empty:
        return float("nan")
    mu, sigma = rp.mean(), rp.std(ddof=ddof)
    if np.isnan(mu) or np.isnan(sigma):
        return float("nan")
    if ann_factor is not None:
        mu = mu * ann_factor
        sigma = sigma * np.sqrt(ann_factor)
    z = norm.ppf(1 - alpha)
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    es = -(mu - sigma * (phi / (1 - alpha)))
    return float(max(es, 0.0))


# Benchmark-based metrics.
def tracking_error(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return tracking_error_from_series(
        portfolio_return_series(returns, weights),
        benchmark,
        ann_factor=ann_factor,
        ddof=ddof,
    )


def tracking_error_from_series(
    portfolio_returns: pd.Series,
    benchmark: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    rp = portfolio_returns.dropna()
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan")
    diff = joint.iloc[:, 0] - joint.iloc[:, 1]
    te = diff.std(ddof=ddof)
    return te if ann_factor is None else te * np.sqrt(ann_factor)


def alpha_beta(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark: pd.Series,
    *,
    rf_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> Tuple[float, float]:
    return alpha_beta_from_series(
        portfolio_return_series(returns, weights),
        benchmark,
        rf_per_period=rf_per_period,
        ann_factor=ann_factor,
        ddof=ddof,
    )


def alpha_beta_from_series(
    portfolio_returns: pd.Series,
    benchmark: pd.Series,
    *,
    rf_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> Tuple[float, float]:
    rp = portfolio_returns.dropna()
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan"), float("nan")
    y = (joint.iloc[:, 0] - rf_per_period).values
    x = (joint.iloc[:, 1] - rf_per_period).values
    if x.std(ddof=ddof) == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(x, y, ddof=ddof)[0, 1] / np.var(x, ddof=ddof))
    alpha = float(y.mean() - beta * x.mean())
    if ann_factor is not None:
        alpha *= ann_factor
    return alpha, beta


def information_ratio(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return information_ratio_from_series(
        portfolio_return_series(returns, weights),
        benchmark,
        ann_factor=ann_factor,
        ddof=ddof,
    )


def information_ratio_from_series(
    portfolio_returns: pd.Series,
    benchmark: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    rp = portfolio_returns.dropna()
    joint = pd.concat([rp, benchmark], axis=1).dropna()
    if joint.shape[1] < 2 or joint.empty:
        return float("nan")
    diff = joint.iloc[:, 0] - joint.iloc[:, 1]
    mu, sd = diff.mean(), diff.std(ddof=ddof)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    ir = mu / sd
    return ir if ann_factor is None else ir * np.sqrt(ann_factor)


# ============================================================
# 3) OPTIONAL WRAPPERS USING FULL RETURN MATRICES
#    (useful when you do not want to assemble mu/cov by hand)
# ============================================================


def expected_return(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
) -> float:
    mu = returns.mean()
    return expected_return_from_moments(mu, weights, ann_factor=ann_factor)


def expected_volatility(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    cov = returns.cov(ddof=ddof)
    return expected_volatility_from_moments(cov, weights, ann_factor=ann_factor)


def sharpe(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    rf_per_period: float = 0.0,
    ann_factor: Optional[int] = None,
    ddof: int = 1,
) -> float:
    return sharpe_from_series(
        portfolio_return_series(returns, weights),
        rf_per_period=rf_per_period,
        ann_factor=ann_factor,
        ddof=ddof,
    )
