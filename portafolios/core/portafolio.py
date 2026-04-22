from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ..data.base import StandardizedData
from ..metrics import asset as am
from ..metrics import portfolio as pm
from ..plots import bar, bubble, corr_heatmap, pie
from ..plots.hrp_plots import getmatrix, histogramadedist, matrizdedist
from .types import ConstructionResult, HRPDiagnostics


class _HRPPlotAdapter:
    def __init__(self, diagnostics: HRPDiagnostics) -> None:
        self.last_dist = diagnostics.distance_matrix
        self.last_clusters = diagnostics.clusters


class PortfolioUniverse:
    """
    Main framework object.

    It keeps a shared dataset and multiple constructions over the same universe.
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
        self.covariance_lr = None
        self.correlation = None
        self.correlation_lr = None

        self.weights: pd.Series | None = None
        self.constructions: dict[str, ConstructionResult] = {}
        self.active_label: str | None = None

        # Backward-compatible alias
        self.info = self.metadata

    def preparar_datos(self) -> "PortfolioUniverse":
        if self.loader is None:
            raise ValueError("You must provide a loader or a standardized data object.")

        data = self._resolve_market_data()

        if not isinstance(data.prices.index, pd.DatetimeIndex):
            raise TypeError("The loader must return prices indexed by a DatetimeIndex.")
        if data.prices.empty:
            raise ValueError("The loader returned an empty DataFrame.")

        self.market_data = data
        self.prices = data.prices.sort_index()
        self.returns = data.returns.sort_index()
        self._reset_construction_state()

        if self.tickers is None:
            self.tickers = data.tickers

        self.asset_returns = self.returns
        self.asset_log_returns = am.returns_log(self.prices)
        self.asset_vol = am.volatility(self.asset_returns)
        self.asset_vol_lr = am.volatility(self.asset_log_returns)
        self.asset_mean_ret = am.mean_return(self.asset_returns)
        self.asset_mean_lr = am.mean_return(self.asset_log_returns)
        # Core portfolio moments use simple returns across the library.
        self.covariance = am.covariance_matrix(self.asset_returns)
        self.correlation = am.correlation_matrix(self.asset_returns)

        # Keep log-return diagnostics available as explicit alternate views.
        self.covariance_lr = am.covariance_matrix(self.asset_log_returns)
        self.correlation_lr = am.correlation_matrix(self.asset_log_returns)

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

    def _reset_construction_state(self) -> None:
        self.weights = None
        self.active_label = None
        self.constructions.clear()
        self.metadata.pop("active_construction", None)
        self.metadata.pop("constructor", None)
        self.metadata.pop("constructor_display_name", None)
        self._last_constructor = None

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
            raise RuntimeError("Prepare the market data before requesting returns.")

        window = self.returns.copy()
        if start is not None:
            window = window.loc[window.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        if dropna:
            window = window.dropna(axis=0, how="any")
        if window.empty:
            raise ValueError("No returns are available inside the requested window.")
        return window

    def _resolve_metric_context(
        self,
        *,
        construction_name: str | None = None,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        if self.asset_returns is None:
            raise RuntimeError("Prepare the market data before requesting metrics.")

        construction = self.get_construction(construction_name or self.active_label) if (construction_name or self.active_label) else None

        if construction is not None:
            returns_frame = self.get_returns_window(
                construction.construction_start,
                construction.construction_end,
            )
            weights = construction.weights.copy()
        elif self.weights is not None:
            returns_frame = self.asset_returns
            weights = self.weights.copy()
        else:
            raise RuntimeError("Build a portfolio before requesting weight-based metrics.")

        weights = weights.reindex(returns_frame.columns).fillna(0.0)
        mean_returns = returns_frame.mean()
        covariance = returns_frame.cov()
        return weights, returns_frame, mean_returns, covariance

    def _resolve_path_metric_series(
        self,
        *,
        construction_name: str | None = None,
    ) -> pd.Series:
        target_name = construction_name or self.active_label
        construction = self.get_construction(target_name) if target_name is not None else None

        if construction is not None and construction.backtest_result is not None:
            return construction.backtest_result.portfolio_returns.copy()

        weights, returns_frame, _, _ = self._resolve_metric_context(construction_name=construction_name)
        return pm.portfolio_return_series(returns_frame, weights)

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

        self._save_method_specific_construction_artifacts(construction, construction_dir)
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
        if backtest.drawdown_series is not None:
            backtest.drawdown_series.to_csv(backtest_dir / "drawdown_series.csv", header=["drawdown"])

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
            "estimation_start": str(mc_result.estimation_start) if mc_result.estimation_start is not None else None,
            "estimation_end": str(mc_result.estimation_end) if mc_result.estimation_end is not None else None,
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

        raise TypeError("The loader must return StandardizedData or a price DataFrame.")

    def construir(self, constructor, label: str | None = None, set_active: bool = True, **kwargs) -> "PortfolioUniverse":
        if self.asset_returns is None:
            raise RuntimeError("Run preparar_datos() before building constructions.")

        result = self._build_construction_result(constructor=constructor, label=label, **kwargs)
        self.add_construction(result, set_active=set_active)
        self._last_constructor = constructor
        return self

    def _extract_construction_diagnostics(self, constructor) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}

        method_id = getattr(constructor, "method_id", "")
        if method_id.startswith("hrp_") and hasattr(constructor, "last_dist"):
            local_weights = {
                cluster_name: cluster_weights.copy()
                for cluster_name, cluster_weights in getattr(constructor, "last_local_weights", {}).items()
            }
            diagnostics["hrp"] = HRPDiagnostics(
                distance_matrix=constructor.last_dist.copy(),
                clusters=[list(cluster) for cluster in getattr(constructor, "last_clusters", [])],
                cluster_returns=getattr(constructor, "last_cluster_returns", pd.DataFrame()).copy(),
                cluster_weights=getattr(constructor, "last_w_clusters", pd.Series(dtype=float)).copy(),
                local_weights=local_weights,
                final_weights=getattr(constructor, "last_final_weights", pd.Series(dtype=float)).copy(),
                linkage_matrix=(
                    getattr(constructor, "last_linkage_matrix", None).copy()
                    if getattr(constructor, "last_linkage_matrix", None) is not None
                    else None
                ),
                cluster_labels=(
                    getattr(constructor, "last_cluster_labels", None).copy()
                    if getattr(constructor, "last_cluster_labels", None) is not None
                    else None
                ),
                cluster_assets={
                    cluster_name: list(assets)
                    for cluster_name, assets in getattr(constructor, "last_cluster_assets", {}).items()
                },
                inner_metadata_by_cluster={
                    cluster_name: dict(meta)
                    for cluster_name, meta in getattr(constructor, "last_inner_meta_by_cluster", {}).items()
                },
                outer_metadata=dict(getattr(constructor, "last_outer_meta", {}) or {}),
                distance_name=getattr(constructor, "distance_name", None),
                clustering_name=getattr(constructor, "clustering_name", None),
            )

        return diagnostics

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
            diagnostics = dict(result.diagnostics)
            diagnostics.update(self._extract_construction_diagnostics(constructor))
            overrides: dict[str, Any] = {}
            if result.construction_start is None:
                overrides["construction_start"] = construction_start
            if result.construction_end is None:
                overrides["construction_end"] = construction_end
            if diagnostics:
                overrides["diagnostics"] = diagnostics
            if overrides:
                return replace(result, **overrides)
            return result

        if not hasattr(constructor, "optimizar"):
            raise TypeError("The constructor must implement `build(...)` or `optimizar(...)`.")

        weights, meta = constructor.optimizar(construction_returns, **kwargs)
        method_id = getattr(constructor, "method_id", type(constructor).__name__)
        display_name = getattr(constructor, "display_name", getattr(constructor, "nombre", method_id))
        name = label or self._next_label(method_id)
        weights = weights.reindex(construction_returns.columns).fillna(0.0).sort_index()
        metrics = self._make_basic_metrics(
            weights,
            returns=construction_returns,
            ann_factor=kwargs.get("ann_factor"),
            rf_per_period=kwargs.get("rf_per_period", getattr(constructor, "rf_per_period", 0.0)),
        )
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
            diagnostics=self._extract_construction_diagnostics(constructor),
        )

    def add_construction(self, result: ConstructionResult, *, set_active: bool = True) -> None:
        normalized = self._normalize_construction_result(result)
        if normalized.name in self.constructions:
            raise ValueError(f"A construction named '{normalized.name}' already exists.")

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
            raise KeyError(f"No construction named '{name}' exists.") from exc

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
        rf_per_period: float = 0.0,
    ) -> dict[str, float]:
        returns_frame = returns if returns is not None else self.asset_returns
        weights = weights.reindex(returns_frame.columns).fillna(0.0)
        mean_returns = returns_frame.mean()
        covariance = returns_frame.cov()

        return {
            "n_selected": int((weights != 0).sum()),
            "expected_return": float(pm.expected_return_from_moments(mean_returns, weights, ann_factor=ann_factor)),
            "volatility": float(pm.expected_volatility_from_moments(covariance, weights, ann_factor=ann_factor)),
            "sharpe_m": float(
                pm.sharpe_from_moments(
                    mean_returns,
                    covariance,
                    weights,
                    rf_per_period=rf_per_period,
                    ann_factor=ann_factor,
                )
            ),
        }

    def make_basic_metrics(
        self,
        weights: pd.Series,
        *,
        returns: pd.DataFrame | None = None,
        ann_factor: int | None = None,
        rf_per_period: float = 0.0,
    ) -> dict[str, float]:
        return self._make_basic_metrics(
            weights,
            returns=returns,
            ann_factor=ann_factor,
            rf_per_period=rf_per_period,
        )

    def _normalize_construction_result(self, result: ConstructionResult) -> ConstructionResult:
        weights = result.weights.reindex(self.asset_returns.columns).fillna(0.0).sort_index()
        selected = [asset for asset in weights.index if weights.loc[asset] != 0]
        return replace(
            result,
            weights=weights,
            selected_assets=selected,
            params=dict(result.params or {}),
            metrics_insample=dict(result.metrics_insample or {}),
            diagnostics=dict(result.diagnostics or {}),
        )

    def _next_label(self, base: str) -> str:
        label = base
        i = 0
        while label in self.constructions:
            i += 1
            label = f"{base}_{i}"
        return label

    def _save_method_specific_construction_artifacts(
        self,
        construction: ConstructionResult,
        construction_dir: Path,
    ) -> None:
        diagnostics_root = construction_dir / "diagnostics"
        shutil.rmtree(diagnostics_root, ignore_errors=True)

        if construction.hrp_diagnostics is not None:
            self._save_hrp_diagnostics(construction, diagnostics_root / "hrp")

        if self._is_markowitz_construction(construction):
            self._save_markowitz_diagnostics(construction, diagnostics_root / "markowitz")

    def _save_hrp_diagnostics(
        self,
        construction: ConstructionResult,
        diagnostics_dir: Path,
    ) -> None:
        diagnostics = construction.hrp_diagnostics
        if diagnostics is None:
            return

        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "construction_name": construction.name,
            "display_name": construction.display_name,
            "distance_name": diagnostics.distance_name,
            "clustering_name": diagnostics.clustering_name,
            "n_clusters": len(diagnostics.clusters),
            "clusters": diagnostics.clusters,
        }
        with (diagnostics_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, default=str)

        diagnostics.distance_matrix.to_csv(diagnostics_dir / "distance_matrix.csv")
        diagnostics.final_weights.to_csv(diagnostics_dir / "final_weights.csv", header=["weight"])

        if not diagnostics.cluster_returns.empty:
            diagnostics.cluster_returns.to_csv(diagnostics_dir / "cluster_returns.csv")

        if not diagnostics.cluster_weights.empty:
            diagnostics.cluster_weights.to_csv(diagnostics_dir / "cluster_weights.csv", header=["weight"])

        local_weights_frame = pd.DataFrame(
            {
                cluster_name: cluster_weights
                for cluster_name, cluster_weights in diagnostics.local_weights.items()
            }
        )
        if not local_weights_frame.empty:
            local_weights_frame.to_csv(diagnostics_dir / "local_weights.csv")

        if diagnostics.cluster_labels is not None:
            diagnostics.cluster_labels.rename("cluster").to_csv(diagnostics_dir / "cluster_labels.csv")

        if diagnostics.linkage_matrix is not None:
            pd.DataFrame(
                diagnostics.linkage_matrix,
                columns=["left", "right", "distance", "count"],
            ).to_csv(diagnostics_dir / "linkage_matrix.csv", index=False)

        with (diagnostics_dir / "cluster_assets.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics.cluster_assets, handle, indent=2, default=str)

        with (diagnostics_dir / "inner_metadata_by_cluster.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics.inner_metadata_by_cluster, handle, indent=2, default=str)

        with (diagnostics_dir / "outer_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics.outer_metadata, handle, indent=2, default=str)

    def _save_markowitz_diagnostics(
        self,
        construction: ConstructionResult,
        diagnostics_dir: Path,
    ) -> None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        returns_frame = self.get_returns_window(
            construction.construction_start,
            construction.construction_end,
        ).dropna(axis=0, how="any")
        returns_frame = returns_frame.loc[:, construction.weights.index]

        expected_returns = returns_frame.mean()
        covariance = returns_frame.cov()
        portfolio_weights = construction.weights.reindex(expected_returns.index).fillna(0.0).astype(float)
        covariance_values = covariance.to_numpy(dtype=float)
        weight_values = portfolio_weights.to_numpy(dtype=float)

        optimizer_meta = {
            key.removeprefix("meta_"): value
            for key, value in (construction.metrics_insample or {}).items()
            if key.startswith("meta_")
        }
        optimizer_meta.setdefault("allow_short", bool(construction.params.get("allow_short", False)))
        optimizer_meta.setdefault("ret_kind_used", construction.params.get("ret_kind", "simple"))

        with (diagnostics_dir / "optimizer_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(optimizer_meta, handle, indent=2, default=str)

        expected_returns.rename("expected_return").to_csv(diagnostics_dir / "expected_returns.csv")
        covariance.to_csv(diagnostics_dir / "covariance.csv")

        asset_moments = pd.DataFrame(
            {
                "expected_return": expected_returns,
                "volatility": pd.Series(
                    np.sqrt(np.diag(covariance_values)),
                    index=expected_returns.index,
                ),
            }
        )
        asset_moments.to_csv(diagnostics_dir / "asset_moments.csv")

        portfolio_point = {
            "construction_name": construction.name,
            "expected_return": float(weight_values @ expected_returns.to_numpy(dtype=float)),
            "volatility": float(np.sqrt(max(float(weight_values @ covariance_values @ weight_values), 0.0))),
            "allow_short": bool(optimizer_meta.get("allow_short", False)),
        }
        with (diagnostics_dir / "portfolio_point.json").open("w", encoding="utf-8") as handle:
            json.dump(portfolio_point, handle, indent=2, default=str)

        from ..plots.visualizer import PortfolioVisualizer

        frontier = PortfolioVisualizer(self)._compute_efficient_frontier(
            expected_returns=expected_returns,
            covariance=covariance,
            allow_short=bool(optimizer_meta.get("allow_short", False)),
            n_points=30,
        )
        frontier.to_csv(diagnostics_dir / "efficient_frontier_points.csv", index=False)

    def _is_markowitz_construction(self, construction: ConstructionResult) -> bool:
        values = (
            str(construction.method_id).lower(),
            str(construction.name).lower(),
            str(construction.display_name).lower(),
        )
        return any("markowitz" in value for value in values)

    def kpi(self, nombre: str, **kwargs):
        ann = kwargs.get("ann_factor", None)
        rf = kwargs.get("rf_per_period", 0.0)
        construction_name = kwargs.get("construction_name")
        weights, returns_frame, mean_returns, covariance = self._resolve_metric_context(
            construction_name=construction_name
        )

        if nombre in ("exp_return", "er"):
            return pm.expected_return_from_moments(mean_returns, weights, ann_factor=ann)
        if nombre in ("vol", "volatility"):
            return pm.expected_volatility_from_moments(covariance, weights, ann_factor=ann)
        if nombre in ("sharpe_m", "sharpe_moments"):
            return pm.sharpe_from_moments(
                mean_returns,
                covariance,
                weights,
                rf_per_period=rf,
                ann_factor=ann,
            )
        if nombre in ("rc", "risk_contrib"):
            return pm.risk_contributions_from_cov(
                covariance,
                weights,
                ann_factor=ann,
                as_fraction=kwargs.get("as_fraction", True),
            )

        portfolio_returns = self._resolve_path_metric_series(construction_name=construction_name)
        if nombre in ("sharpe",):
            return pm.sharpe_from_series(portfolio_returns, rf_per_period=rf, ann_factor=ann)
        if nombre in ("sortino",):
            return pm.sortino_from_series(
                portfolio_returns,
                mar_per_period=kwargs.get("mar_per_period", 0.0),
                ann_factor=ann,
            )
        if nombre in ("mdd", "max_drawdown"):
            return pm.max_drawdown_from_series(portfolio_returns)
        if nombre in ("var", "var_gauss"):
            return pm.var_gaussian_from_series(
                portfolio_returns,
                alpha=kwargs.get("alpha", 0.95),
                ann_factor=ann,
            )
        if nombre in ("cvar", "cvar_gauss", "es"):
            return pm.cvar_gaussian_from_series(
                portfolio_returns,
                alpha=kwargs.get("alpha", 0.95),
                ann_factor=ann,
            )
        if nombre in ("te", "tracking_error"):
            bench = kwargs["benchmark"]
            return pm.tracking_error_from_series(portfolio_returns, bench, ann_factor=ann)
        if nombre in ("alpha_beta", "ab"):
            bench = kwargs["benchmark"]
            return pm.alpha_beta_from_series(portfolio_returns, bench, rf_per_period=rf, ann_factor=ann)
        if nombre in ("ir", "information_ratio"):
            bench = kwargs["benchmark"]
            return pm.information_ratio_from_series(portfolio_returns, bench, ann_factor=ann)

        raise ValueError(f"Unrecognized KPI: {nombre}")

    def get_metric(self, metric_name: str, **kwargs):
        return self.kpi(metric_name, **kwargs)

    def get_basic_metrics(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        return self.kpis_basicos(ann_factor=ann_factor, rf_per_period=rf_per_period)

    def kpis_basicos(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        weights, returns_frame, _, _ = self._resolve_metric_context()
        metrics = self._make_basic_metrics(
            weights,
            returns=returns_frame,
            ann_factor=ann_factor,
            rf_per_period=rf_per_period,
        )
        return {
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe_m": metrics["sharpe_m"],
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

    def _get_hrp_diagnostics(self, construction_name: str | None = None) -> Optional[HRPDiagnostics]:
        target_name = construction_name or self.active_label
        if target_name is None:
            print("No active construction is selected. Build one first or pass construction_name.")
            return None

        construction = self.get_construction(target_name)
        diagnostics = construction.hrp_diagnostics
        if diagnostics is None:
            print(f"Construction '{construction.name}' does not include HRP diagnostics.")
            return None

        return diagnostics

    def plot_hrp_matriz_distancias(
        self,
        file_path: str | None = None,
        construction_name: str | None = None,
    ):
        diagnostics = self._get_hrp_diagnostics(construction_name=construction_name)
        if diagnostics is None:
            return None

        return matrizdedist.matriz_distancias(_HRPPlotAdapter(diagnostics), file_path=file_path)

    def plot_hrp_hist_distancias(
        self,
        file_path: str | None = "hrp_hist_distancias_deprado.html",
        construction_name: str | None = None,
    ) -> None:
        diagnostics = self._get_hrp_diagnostics(construction_name=construction_name)
        if diagnostics is None:
            return None

        return histogramadedist.histograma_distancias(_HRPPlotAdapter(diagnostics), file_path=file_path)

    def get_hrp_dist_matrix(self, construction_name: str | None = None) -> Optional[pd.DataFrame]:
        diagnostics = self._get_hrp_diagnostics(construction_name=construction_name)
        if diagnostics is None:
            return None

        return getmatrix.get_distmat(_HRPPlotAdapter(diagnostics))


Portfolio = PortfolioUniverse
Universe = PortfolioUniverse

__all__ = ["PortfolioUniverse", "Universe", "Portfolio"]
