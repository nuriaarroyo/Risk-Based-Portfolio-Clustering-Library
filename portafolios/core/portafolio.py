# portafolios/core/portafolio.py
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import pandas as pd

from portafolios.constructores.hrp_style.hrp_core import HRPStyle

from ..data.base import StandardizedData
from ..metrics import asset as am
from ..metrics import portfolio as pm
from ..plots import bar, bubble, corr_heatmap, pie
from ..plots.hrp_plots import getmatrix, histogramadedist, matrizdedist
from .types import ConstructionResult


class Portfolio:
    """
    Objeto principal del framework.

    Mantiene el dataset compartido y multiples construcciones sobre el mismo universo.
    """

    def __init__(
        self,
        *,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        loader: Optional[Callable[..., pd.DataFrame] | Any] = None,
        loader_kwargs: Optional[dict[str, Any]] = None,
        freq: Optional[str] = "B",
    ):
        self.tickers = list(tickers) if tickers is not None else None
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.freq = freq
        self.loader = loader
        self.loader_kwargs = loader_kwargs or {}

        self.market_data: Optional[StandardizedData] = None
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.metadata: dict[str, Any] = {}

        self.asset_returns = None
        self.asset_log_returns = None
        self.asset_vol = None
        self.asset_vol_lr = None
        self.asset_mean_ret = None
        self.asset_mean_lr = None
        self.covariance = None
        self.correlation = None

        self.weights: pd.Series | None = None
        self.constructions: dict[str, ConstructionResult] = {}
        self.active_label: str | None = None

        # Backward-compatible alias
        self.info = self.metadata

    def preparar_datos(self) -> "Portfolio":
        if self.loader is None:
            raise ValueError("Debes proporcionar un loader o un objeto de datos estandarizado.")

        data = self._resolve_market_data()

        if not isinstance(data.prices.index, pd.DatetimeIndex):
            raise TypeError("El loader debe devolver precios con DatetimeIndex.")
        if data.prices.empty:
            raise ValueError("El loader devolvio un DataFrame vacio.")

        self.market_data = data
        self.prices = data.prices.sort_index()
        self.returns = data.returns.sort_index()

        if self.tickers is None:
            self.tickers = data.tickers

        self.asset_returns = self.returns
        self.asset_log_returns = am.returns_log(self.prices)
        self.asset_vol = am.volatility(self.asset_returns)
        self.asset_vol_lr = am.volatility(self.asset_log_returns)
        self.asset_mean_ret = am.mean_return(self.asset_returns)
        self.asset_mean_lr = am.mean_return(self.asset_log_returns)
        self.covariance = am.covariance_matrix(self.asset_log_returns)
        self.correlation = am.correlation_matrix(self.asset_log_returns)

        self.metadata.clear()
        self.metadata.update(data.metadata)

        return self

    def _resolve_market_data(self) -> StandardizedData:
        if isinstance(self.loader, StandardizedData):
            return self.loader

        if hasattr(self.loader, "get_data"):
            return self._coerce_market_data(self.loader.get_data())

        raw = self.loader(
            tickers=self.tickers,
            start=(self.start.isoformat() if self.start is not None else None),
            end=(self.end.isoformat() if self.end is not None else None),
            freq=self.freq,
            **self.loader_kwargs,
        )
        return self._coerce_market_data(raw)

    def _coerce_market_data(self, raw: Any) -> StandardizedData:
        if isinstance(raw, StandardizedData):
            return raw

        if isinstance(raw, pd.DataFrame):
            prices = raw.sort_index()
            return StandardizedData(
                prices=prices,
                returns=am.returns_simple(prices),
                tickers=list(prices.columns),
                metadata={"source": "callable_dataframe", "frequency": self.freq},
            )

        raise TypeError("El loader debe devolver StandardizedData o un DataFrame de precios.")

    def construir(self, constructor, label: str | None = None, set_active: bool = True, **kwargs) -> "Portfolio":
        if self.asset_returns is None:
            raise RuntimeError("Llama primero a preparar_datos().")

        result = self._build_construction_result(constructor=constructor, label=label, **kwargs)
        self.add_construction(result, set_active=set_active)
        self._last_constructor = constructor
        return self

    def _build_construction_result(self, constructor, *, label: str | None = None, **kwargs) -> ConstructionResult:
        if hasattr(constructor, "build"):
            name = label or self._next_label(getattr(constructor, "nombre", type(constructor).__name__))
            result = constructor.build(self, name=name, **kwargs)
            return self._normalize_construction_result(result)

        if not hasattr(constructor, "optimizar"):
            raise TypeError("El constructor debe implementar `build(...)` o `optimizar(...)`.")

        weights, meta = constructor.optimizar(self.asset_returns, **kwargs)
        method = getattr(constructor, "nombre", type(constructor).__name__)
        name = label or self._next_label(method)
        weights = weights.reindex(self.asset_returns.columns).fillna(0.0).sort_index()
        metrics = self._make_basic_metrics(weights, ann_factor=kwargs.get("ann_factor"))
        if meta:
            metrics.update({f"meta_{key}": value for key, value in meta.items()})

        return ConstructionResult(
            name=name,
            method=method,
            weights=weights,
            selected_assets=[asset for asset in weights.index if weights.loc[asset] != 0],
            params=dict(kwargs),
            metrics_insample=metrics,
            backtest_result=None,
            mc_result=None,
            notes=kwargs.get("notes"),
        )

    def add_construction(self, result: ConstructionResult, *, set_active: bool = True) -> None:
        normalized = self._normalize_construction_result(result)
        if normalized.name in self.constructions:
            raise ValueError(f"Ya existe una construccion con el nombre '{normalized.name}'.")

        self.constructions[normalized.name] = normalized

        if set_active:
            self.weights = normalized.weights.copy()
            self.active_label = normalized.name
            self.metadata.update(
                {
                    "active_construction": normalized.name,
                    "constructor": normalized.method,
                }
            )

    def get_construction(self, name: str) -> ConstructionResult:
        try:
            return self.constructions[name]
        except KeyError as exc:
            raise KeyError(f"No existe una construccion llamada '{name}'.") from exc

    def list_constructions(self) -> list[str]:
        return list(self.constructions.keys())

    def compare_insample_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for name, result in self.constructions.items():
            row = {"name": name, "method": result.method}
            row.update(result.metrics_insample)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["name", "method"]).set_index("name")

        return pd.DataFrame(rows).set_index("name").sort_index()

    def _make_basic_metrics(self, weights: pd.Series, *, ann_factor: int | None = None) -> dict[str, float]:
        weights = weights.reindex(self.asset_returns.columns).fillna(0.0)
        mean_returns = self.asset_returns.mean()
        covariance = self.asset_returns.cov()

        return {
            "n_selected": int((weights != 0).sum()),
            "expected_return": float(pm.expected_return_from_moments(mean_returns, weights, ann_factor=ann_factor)),
            "volatility": float(pm.expected_volatility_from_moments(covariance, weights, ann_factor=ann_factor)),
            "sharpe_m": float(pm.sharpe_from_moments(mean_returns, covariance, weights, ann_factor=ann_factor)),
        }

    def make_basic_metrics(self, weights: pd.Series, *, ann_factor: int | None = None) -> dict[str, float]:
        return self._make_basic_metrics(weights, ann_factor=ann_factor)

    def _normalize_construction_result(self, result: ConstructionResult) -> ConstructionResult:
        weights = result.weights.reindex(self.asset_returns.columns).fillna(0.0).sort_index()
        selected = [asset for asset in weights.index if weights.loc[asset] != 0]
        return ConstructionResult(
            name=result.name,
            method=result.method,
            weights=weights,
            selected_assets=selected,
            params=dict(result.params),
            metrics_insample=dict(result.metrics_insample),
            backtest_result=result.backtest_result,
            mc_result=result.mc_result,
            notes=result.notes,
        )

    def _next_label(self, base: str) -> str:
        label = base
        i = 0
        while label in self.constructions:
            i += 1
            label = f"{base}_{i}"
        return label

    def kpi(self, nombre: str, **kwargs):
        if self.weights is None:
            raise RuntimeError("Primero construye el portafolio para tener weights.")

        ann = kwargs.get("ann_factor", None)
        rf = kwargs.get("rf_per_period", 0.0)

        if nombre in ("exp_return", "er"):
            return pm.expected_return_from_moments(self.asset_mean_ret, self.weights, ann_factor=ann)
        if nombre in ("vol", "volatility"):
            return pm.expected_volatility_from_moments(self.covariance, self.weights, ann_factor=ann)
        if nombre in ("sharpe_m", "sharpe_moments"):
            return pm.sharpe_from_moments(
                self.asset_mean_ret,
                self.covariance,
                self.weights,
                rf_per_period=rf,
                ann_factor=ann,
            )
        if nombre in ("rc", "risk_contrib"):
            return pm.risk_contributions_from_cov(
                self.covariance,
                self.weights,
                ann_factor=ann,
                as_fraction=kwargs.get("as_fraction", True),
            )

        if nombre in ("sharpe",):
            return pm.sharpe(self.asset_returns, self.weights, rf_per_period=rf, ann_factor=ann)
        if nombre in ("sortino",):
            return pm.sortino(
                self.asset_returns,
                self.weights,
                mar_per_period=kwargs.get("mar_per_period", 0.0),
                ann_factor=ann,
            )
        if nombre in ("mdd", "max_drawdown"):
            return pm.max_drawdown(self.asset_returns, self.weights)
        if nombre in ("calmar",):
            return pm.calmar_from_moments(
                self.asset_mean_ret,
                self.covariance,
                self.weights,
                ann_factor=ann,
                returns_for_mdd=self.asset_returns,
            )
        if nombre in ("var", "var_gauss"):
            return pm.var_gaussian(self.asset_returns, self.weights, alpha=kwargs.get("alpha", 0.95), ann_factor=ann)
        if nombre in ("cvar", "cvar_gauss", "es"):
            return pm.cvar_gaussian(
                self.asset_returns,
                self.weights,
                alpha=kwargs.get("alpha", 0.95),
                ann_factor=ann,
            )
        if nombre in ("te", "tracking_error"):
            bench = kwargs["benchmark"]
            return pm.tracking_error(self.asset_returns, self.weights, bench, ann_factor=ann)
        if nombre in ("alpha_beta", "ab"):
            bench = kwargs["benchmark"]
            return pm.alpha_beta(self.asset_returns, self.weights, bench, rf_per_period=rf, ann_factor=ann)
        if nombre in ("ir", "information_ratio"):
            bench = kwargs["benchmark"]
            return pm.information_ratio(self.asset_returns, self.weights, bench, ann_factor=ann)

        raise ValueError(f"KPI no reconocido: {nombre}")

    def kpis_basicos(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        return {
            "expected_return": self.kpi("exp_return", ann_factor=ann_factor),
            "volatility": self.kpi("vol", ann_factor=ann_factor),
            "sharpe_m": self.kpi("sharpe_m", ann_factor=ann_factor, rf_per_period=rf_per_period),
        }

    def pastel(self, pesos=None, min_weight: float = 1e-3) -> None:
        return pie.pastel_portfolio(self, pesos=pesos, min_weight=min_weight)

    def barras(self, pesos=None, min_weight: float = 0.0) -> None:
        return bar.barras_portfolio(self, pesos=pesos, min_weight=min_weight)

    def bubbleplot(self, pesos=None, min_weight: float = 0.0):
        return bubble.bubbleplot_portfolio(self, pesos=pesos, min_weight=min_weight)

    def corr_heatmap(self, kind: str = "correlation", round_decimals: int = 2) -> None:
        return corr_heatmap.corr_heatmap_portfolio(self, kind=kind, round_decimals=round_decimals)

    def plot_hrp_matriz_distancias(self, file_path: str | None = None):
        ctor = getattr(self, "_last_constructor", None)

        if ctor is None:
            print("Aun no has construido el portafolio con ningun constructor.")
            return

        if not hasattr(ctor, "last_dist"):
            print(
                "Funcion disponible solo si el portafolio fue construido con HRPStyle "
                "y ya se corrio al menos una vez."
            )
            return

        return matrizdedist.matriz_distancias(ctor, file_path=file_path)

    def plot_hrp_hist_distancias(self) -> None:
        ctor = getattr(self, "_last_constructor", None)

        if ctor is None:
            print("Aun no has construido el portafolio con ningun constructor.")
            return

        if not isinstance(ctor, HRPStyle):
            print("Funcion disponible solo si el portafolio fue construido con HRPStyle.")
            return

        if not hasattr(ctor, "last_dist"):
            print("Este HRPStyle no tiene datos de distancias. Ya corriste construir?")
            return

        histogramadedist.histograma_distancias(ctor)

    def get_hrp_dist_matrix(self) -> Optional[pd.DataFrame]:
        ctor = getattr(self, "_last_constructor", None)

        if ctor is None:
            print("Aun no has construido el portafolio con ningun constructor.")
            return None

        if not isinstance(ctor, HRPStyle):
            print("Funcion disponible solo si el portafolio fue construido con HRPStyle.")
            return None

        if not hasattr(ctor, "last_dist"):
            print("Este HRPStyle no tiene datos de distancias. Ya corriste construir?")
            return None

        return getmatrix.get_distmat(ctor)


Universe = Portfolio
