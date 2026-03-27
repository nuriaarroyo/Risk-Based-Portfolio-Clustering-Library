# portafolios/core/portafolio.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from portafolios.constructores.hrp_style.hrp_core import HRPStyle

from ..data.base import StandardizedData
from ..metrics import asset as am
from ..metrics import portfolio as pm
from ..plots import bar, bubble, corr_heatmap, pie
from ..plots.hrp_plots import getmatrix, histogramadedist, matrizdedist
from .types import ConstructionResult


class PortfolioUniverse:
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
        construction_start: Optional[str] = None,
        construction_end: Optional[str] = None,
        loader: Optional[Callable[..., pd.DataFrame] | Any] = None,
        loader_kwargs: Optional[dict[str, Any]] = None,
        freq: Optional[str] = "B",
        universe_name: Optional[str] = None,
        base_output_dir: str | Path | None = None,
        auto_save_data: bool = True,
    ):
        self.tickers = list(tickers) if tickers is not None else None
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.construction_start = pd.Timestamp(construction_start) if construction_start else self.start
        self.construction_end = pd.Timestamp(construction_end) if construction_end else self.end
        self.freq = freq
        self.loader = loader
        self.loader_kwargs = loader_kwargs or {}
        self.universe_name = universe_name or "universe"
        self.base_output_dir = Path(base_output_dir) if base_output_dir is not None else Path("outputs") / "runs"
        self.auto_save_data = auto_save_data

        self.output_dir = self.base_output_dir / self.universe_name
        self.data_dir = self.output_dir / "data"
        self.constructions_dir = self.output_dir / "constructions"
        self.backtests_dir = self.output_dir / "backtests"
        self.monte_carlo_dir = self.output_dir / "monte_carlo"
        self.plots_dir = self.output_dir / "plots"
        self.create_output_dirs()

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

    def preparar_datos(self) -> "PortfolioUniverse":
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
        self.metadata.update(
            {
                "universe_name": self.universe_name,
                "output_dir": str(self.output_dir),
            }
        )

        if self.auto_save_data:
            self.save_market_data()

        return self

    def create_output_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.data_dir,
            self.constructions_dir,
            self.backtests_dir,
            self.monte_carlo_dir,
            self.plots_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def get_construction_dir(self, construction_name: str) -> Path:
        path = self.constructions_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_backtest_dir(self, construction_name: str) -> Path:
        path = self.backtests_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_mc_dir(self, construction_name: str) -> Path:
        path = self.monte_carlo_dir / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_plot_dir(self, category: str | None = None, construction_name: str | None = None) -> Path:
        path = self.plots_dir
        if category is not None:
            path = path / category
        if construction_name is not None:
            path = path / construction_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_returns_window(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        dropna: bool = True,
    ) -> pd.DataFrame:
        if self.returns is None:
            raise RuntimeError("Primero prepara los datos para tener retornos disponibles.")

        window = self.returns.copy()
        if start is not None:
            window = window.loc[window.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        if dropna:
            window = window.dropna(axis=0, how="any")
        if window.empty:
            raise ValueError("No hay retornos disponibles en la ventana solicitada.")
        return window

    def set_construction_window(
        self,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> "PortfolioUniverse":
        self.construction_start = pd.Timestamp(start) if start is not None else None
        self.construction_end = pd.Timestamp(end) if end is not None else None
        return self

    def save_market_data(self) -> None:
        self.create_output_dirs()

        if self.prices is not None and not self.prices.empty:
            self.prices.to_csv(self.data_dir / "prices.csv")

        if self.returns is not None and not self.returns.empty:
            self.returns.to_csv(self.data_dir / "returns.csv")

        metadata_payload = dict(self.metadata)
        metadata_payload.update(
            {
                "universe_name": self.universe_name,
                "output_dir": str(self.output_dir),
                "frequency": self.freq,
                "start": str(self.start) if self.start is not None else None,
                "end": str(self.end) if self.end is not None else None,
            }
        )

        with (self.data_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, default=str)

        tickers_payload = list(self.tickers) if self.tickers is not None else []
        with (self.data_dir / "tickers.json").open("w", encoding="utf-8") as f:
            json.dump(tickers_payload, f, indent=2)

    def save_construction(self, construction_name: str) -> Path:
        construction = self.get_construction(construction_name)
        construction_dir = self.get_construction_dir(construction_name)

        construction.weights.to_csv(construction_dir / "weights.csv", header=["weight"])

        with (construction_dir / "selected_assets.json").open("w", encoding="utf-8") as f:
            json.dump(construction.selected_assets, f, indent=2)

        with (construction_dir / "params.json").open("w", encoding="utf-8") as f:
            json.dump(construction.params or {}, f, indent=2, default=str)

        with (construction_dir / "metrics_insample.json").open("w", encoding="utf-8") as f:
            json.dump(construction.metrics_insample or {}, f, indent=2, default=str)

        window_payload = {
            "construction_start": str(construction.construction_start) if construction.construction_start is not None else None,
            "construction_end": str(construction.construction_end) if construction.construction_end is not None else None,
        }
        if any(value is not None for value in window_payload.values()):
            with (construction_dir / "construction_window.json").open("w", encoding="utf-8") as f:
                json.dump(window_payload, f, indent=2)

        summary_payload = {
            "name": construction.name,
            "method": construction.method,
            "method_id": construction.method_id,
            "display_name": construction.display_name,
            "notes": construction.notes,
        }
        with (construction_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, default=str)

        return construction_dir

    def save_all_constructions(self) -> dict[str, Path]:
        saved_paths: dict[str, Path] = {}
        for construction_name in self.list_constructions():
            saved_paths[construction_name] = self.save_construction(construction_name)
        return saved_paths

    def save_backtest(self, construction_name: str) -> Path | None:
        construction = self.get_construction(construction_name)
        backtest = construction.backtest_result
        if backtest is None:
            return None

        backtest_dir = self.get_backtest_dir(construction_name)

        backtest.portfolio_returns.to_csv(backtest_dir / "portfolio_returns.csv", header=["return"])
        backtest.cumulative_returns.to_csv(backtest_dir / "cumulative_returns.csv", header=["cumulative_return"])

        with (backtest_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(backtest.summary_metrics or {}, f, indent=2, default=str)

        window_payload = {
            "start_date": str(backtest.start_date),
            "end_date": str(backtest.end_date),
        }
        with (backtest_dir / "backtest_window.json").open("w", encoding="utf-8") as f:
            json.dump(window_payload, f, indent=2, default=str)

        if backtest.notes:
            with (backtest_dir / "notes.txt").open("w", encoding="utf-8") as f:
                f.write(str(backtest.notes))

        return backtest_dir

    def save_all_backtests(self) -> dict[str, Path]:
        saved_paths: dict[str, Path] = {}
        for construction_name in self.list_constructions():
            saved_path = self.save_backtest(construction_name)
            if saved_path is not None:
                saved_paths[construction_name] = saved_path
        return saved_paths

    def save_monte_carlo(self, construction_name: str) -> Path | None:
        construction = self.get_construction(construction_name)
        mc_result = construction.mc_result
        if mc_result is None:
            return None

        mc_dir = self.get_mc_dir(construction_name)

        simulated_paths = mc_result.simulated_paths
        if isinstance(simulated_paths, pd.DataFrame):
            simulated_paths.to_csv(mc_dir / "simulated_paths.csv")
        else:
            pd.DataFrame(simulated_paths).to_csv(mc_dir / "simulated_paths.csv", index=False)

        pd.DataFrame({"terminal_value": mc_result.terminal_values}).to_csv(
            mc_dir / "terminal_values.csv",
            index=False,
        )

        with (mc_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(mc_result.summary_metrics or {}, f, indent=2, default=str)

        config_payload = {
            "construction_name": mc_result.construction_name,
            "horizon": mc_result.horizon,
            "n_simulations": mc_result.n_simulations,
        }
        with (mc_dir / "simulation_config.json").open("w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, default=str)

        if mc_result.notes:
            with (mc_dir / "notes.txt").open("w", encoding="utf-8") as f:
                f.write(str(mc_result.notes))

        return mc_dir

    def save_all_monte_carlo(self) -> dict[str, Path]:
        saved_paths: dict[str, Path] = {}
        for construction_name in self.list_constructions():
            saved_path = self.save_monte_carlo(construction_name)
            if saved_path is not None:
                saved_paths[construction_name] = saved_path
        return saved_paths

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

    def construir(self, constructor, label: str | None = None, set_active: bool = True, **kwargs) -> "PortfolioUniverse":
        if self.asset_returns is None:
            raise RuntimeError("Llama primero a preparar_datos().")

        result = self._build_construction_result(constructor=constructor, label=label, **kwargs)
        self.add_construction(result, set_active=set_active)
        self._last_constructor = constructor
        return self

    def _build_construction_result(self, constructor, *, label: str | None = None, **kwargs) -> ConstructionResult:
        construction_start = (
            pd.Timestamp(kwargs.get("construction_start"))
            if kwargs.get("construction_start") is not None
            else self.construction_start
        )
        construction_end = (
            pd.Timestamp(kwargs.get("construction_end"))
            if kwargs.get("construction_end") is not None
            else self.construction_end
        )
        construction_returns = self.get_returns_window(construction_start, construction_end)

        if hasattr(constructor, "build"):
            method_id = getattr(constructor, "method_id", type(constructor).__name__)
            name = label or self._next_label(method_id)
            result = constructor.build(self, name=name, returns=construction_returns, **kwargs)
            result = self._normalize_construction_result(result)
            if result.construction_start is None or result.construction_end is None:
                return ConstructionResult(
                    name=result.name,
                    method_id=result.method_id,
                    display_name=result.display_name,
                    weights=result.weights,
                    selected_assets=result.selected_assets,
                    params=result.params,
                    metrics_insample=result.metrics_insample,
                    construction_start=construction_start if result.construction_start is None else result.construction_start,
                    construction_end=construction_end if result.construction_end is None else result.construction_end,
                    backtest_result=result.backtest_result,
                    mc_result=result.mc_result,
                    notes=result.notes,
                )
            return result

        if not hasattr(constructor, "optimizar"):
            raise TypeError("El constructor debe implementar `build(...)` o `optimizar(...)`.")

        weights, meta = constructor.optimizar(construction_returns, **kwargs)
        method_id = getattr(constructor, "method_id", type(constructor).__name__)
        display_name = getattr(constructor, "display_name", getattr(constructor, "nombre", method_id))
        name = label or self._next_label(method_id)
        weights = weights.reindex(construction_returns.columns).fillna(0.0).sort_index()
        metrics = self._make_basic_metrics(weights, returns=construction_returns, ann_factor=kwargs.get("ann_factor"))
        if meta:
            metrics.update({f"meta_{key}": value for key, value in meta.items()})

        return ConstructionResult(
            name=name,
            method_id=method_id,
            display_name=display_name,
            weights=weights,
            selected_assets=[asset for asset in weights.index if weights.loc[asset] != 0],
            params=dict(kwargs),
            metrics_insample=metrics,
            construction_start=construction_start,
            construction_end=construction_end,
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
                    "constructor_display_name": normalized.display_name,
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
            row = {
                "name": name,
                "method": result.method,
                "display_name": result.display_name,
            }
            row.update(result.metrics_insample)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["name", "method"]).set_index("name")

        return pd.DataFrame(rows).set_index("name").sort_index()

    def _make_basic_metrics(
        self,
        weights: pd.Series,
        *,
        returns: pd.DataFrame | None = None,
        ann_factor: int | None = None,
    ) -> dict[str, float]:
        returns_frame = returns if returns is not None else self.asset_returns
        weights = weights.reindex(returns_frame.columns).fillna(0.0)
        mean_returns = returns_frame.mean()
        covariance = returns_frame.cov()

        return {
            "n_selected": int((weights != 0).sum()),
            "expected_return": float(pm.expected_return_from_moments(mean_returns, weights, ann_factor=ann_factor)),
            "volatility": float(pm.expected_volatility_from_moments(covariance, weights, ann_factor=ann_factor)),
            "sharpe_m": float(pm.sharpe_from_moments(mean_returns, covariance, weights, ann_factor=ann_factor)),
        }

    def make_basic_metrics(
        self,
        weights: pd.Series,
        *,
        returns: pd.DataFrame | None = None,
        ann_factor: int | None = None,
    ) -> dict[str, float]:
        return self._make_basic_metrics(weights, returns=returns, ann_factor=ann_factor)

    def _normalize_construction_result(self, result: ConstructionResult) -> ConstructionResult:
        weights = result.weights.reindex(self.asset_returns.columns).fillna(0.0).sort_index()
        selected = [asset for asset in weights.index if weights.loc[asset] != 0]
        return ConstructionResult(
            name=result.name,
            method_id=result.method_id,
            display_name=result.display_name,
            weights=weights,
            selected_assets=selected,
            params=dict(result.params),
            metrics_insample=dict(result.metrics_insample),
            construction_start=result.construction_start,
            construction_end=result.construction_end,
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

    def get_metric(self, metric_name: str, **kwargs):
        return self.kpi(metric_name, **kwargs)

    def get_basic_metrics(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        return self.kpis_basicos(ann_factor=ann_factor, rf_per_period=rf_per_period)

    def kpis_basicos(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        return {
            "expected_return": self.kpi("exp_return", ann_factor=ann_factor),
            "volatility": self.kpi("vol", ann_factor=ann_factor),
            "sharpe_m": self.kpi("sharpe_m", ann_factor=ann_factor, rf_per_period=rf_per_period),
        }

    def prepare_data(self) -> "PortfolioUniverse":
        return self.preparar_datos()

    def build(self, constructor, label: str | None = None, set_active: bool = True, **kwargs) -> "PortfolioUniverse":
        return self.construir(constructor, label=label, set_active=set_active, **kwargs)

    def plot_pie(self, weights=None, min_weight: float = 1e-3) -> None:
        return self.pastel(pesos=weights, min_weight=min_weight)

    def plot_bar(self, weights=None, min_weight: float = 0.0) -> None:
        return self.barras(pesos=weights, min_weight=min_weight)

    def plot_bubble(self, weights=None, min_weight: float = 0.0):
        return self.bubbleplot(pesos=weights, min_weight=min_weight)

    def plot_correlation_heatmap(self, kind: str = "correlation", round_decimals: int = 2) -> None:
        return self.corr_heatmap(kind=kind, round_decimals=round_decimals)

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


Portfolio = PortfolioUniverse
Universe = PortfolioUniverse

__all__ = ["PortfolioUniverse", "Universe", "Portfolio"]
