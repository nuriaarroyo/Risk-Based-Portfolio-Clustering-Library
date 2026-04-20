from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portafolios import (  # noqa: E402
    Backtester,
    EqualWeightConstructor,
    HRPRecursive,
    HRPStyle,
    Markowitz,
    MonteCarloEngine,
    NaiveRiskParity,
    PortfolioVisualizer,
    StandardizedData,
    Universe,
    local_loader,
    yfinance_loader,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_experimental_setup"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"
FRAMEWORK_DIR = OUTPUT_DIR / "framework_runs"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_clean_stock_data.csv"
YF_SAVE_PATH = DATA_DIR / "yfinance_final_setup_raw.csv"
YF_CATALOG_PATH = DATA_DIR / "yfinance_snapshot_catalog.json"

ANNUALIZATION_FACTOR = 252
MC_SIMULATIONS = 2_000
MC_INITIAL_VALUE = 1.0
MC_SEED = 42

CONSTRUCTION_DATES = ["2019-06-01", "2020-06-01", "2022-01-03"]
ESTIMATION_WINDOWS_MONTHS = [12, 24, 36]

MULTI_ASSET_ETFS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "TIP",
    "GLD",
    "DBC",
    "VNQ",
]

DJIA_CONSTITUENTS = [
    "AAPL",
    "AMGN",
    "AMZN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "NVDA",
    "PG",
    "SHW",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WMT",
]

# A stable, explicit top-market-cap style candidate list. The runner keeps the
# available subset after loading data, so the experiment is reproducible even
# when the local CSV has fewer than 100 assets.
SP100_STYLE_CANDIDATES = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOG",
    "META",
    "AVGO",
    "TSLA",
    "BRK-B",
    "LLY",
    "JPM",
    "V",
    "WMT",
    "MA",
    "NFLX",
    "ORCL",
    "XOM",
    "COST",
    "JNJ",
    "HD",
    "PG",
    "BAC",
    "ABBV",
    "KO",
    "PM",
    "PLTR",
    "UNH",
    "CRM",
    "CSCO",
    "IBM",
    "WFC",
    "CVX",
    "ABT",
    "MCD",
    "GE",
    "LIN",
    "AMD",
    "MRK",
    "TMO",
    "DIS",
    "PEP",
    "ACN",
    "INTU",
    "ISRG",
    "QCOM",
    "TXN",
    "VZ",
    "AMGN",
    "NOW",
    "ADBE",
    "CAT",
    "GS",
    "UBER",
    "RTX",
    "BKNG",
    "SPGI",
    "PGR",
    "T",
    "NEE",
    "BSX",
    "DHR",
    "PFE",
    "AXP",
    "LOW",
    "UNP",
    "TJX",
    "HON",
    "SYK",
    "SCHW",
    "ETN",
    "BLK",
    "C",
    "GILD",
    "DE",
    "AMAT",
    "BA",
    "PANW",
    "ADP",
    "MDT",
    "VRTX",
    "COP",
    "CB",
    "LRCX",
    "ADI",
    "MMC",
    "MU",
    "SBUX",
    "KLAC",
    "ANET",
    "PLD",
    "MDLZ",
    "LMT",
    "NKE",
    "UPS",
    "SO",
    "MO",
    "ICE",
    "BMY",
    "ELV",
    "WM",
    "DUK",
]


@dataclass(frozen=True)
class UniverseSpec:
    universe_id: str
    display_name: str
    tickers: list[str]


def clean_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    return ticker.lstrip("0") or ticker


def load_processed_ohlcv_close_prices(path: Path, close_field: str = "Close") -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    tickers = raw.iloc[0, 1:].ffill()
    fields = raw.iloc[1, 1:]
    data = raw.iloc[3:].copy()
    dates = pd.to_datetime(data.iloc[:, 0])

    prices: dict[str, object] = {}
    for column_position, (ticker, field) in enumerate(zip(tickers, fields), start=1):
        if str(field).strip().lower() == close_field.lower():
            prices[clean_ticker(ticker)] = pd.to_numeric(
                data.iloc[:, column_position],
                errors="coerce",
            ).to_numpy()

    return pd.DataFrame(prices, index=dates).sort_index().dropna(axis=1, how="all")


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def resolve_local_csv_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_price_source(
    source: str,
    tickers: list[str],
    start: str,
    end: str,
    *,
    local_path: str | Path | None = None,
) -> pd.DataFrame:
    if source == "processed":
        prices = load_processed_ohlcv_close_prices(PROCESSED_DATA_PATH)
        return prices.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    if source == "local":
        if local_path is None:
            raise ValueError("source='local' requiere `local_path` con un CSV estilo yfinance.")
        return local_loader(
            path=resolve_local_csv_path(local_path),
            tickers=tickers,
            start=start,
            end=end,
            prefer_adj_close=True,
        )

    if source == "yfinance":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return yfinance_loader(
            tickers=tickers,
            start=start,
            end=end,
            prefer_adj_close=True,
            save_path=YF_SAVE_PATH,
            use_saved_data=YF_SAVE_PATH.exists(),
            save_download=True,
            catalog_path=YF_CATALOG_PATH,
            fallback_to_saved_data=True,
            batch_size=50,
            progress=False,
        )

    raise ValueError("source must be one of: 'processed', 'local', or 'yfinance'.")


def available_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_assets: int = 2,
) -> pd.DataFrame:
    available = [ticker for ticker in tickers if ticker in prices.columns]
    if not available:
        raise ValueError("None of the requested tickers are available in the price data.")
    window = prices.loc[start:end, available].copy()
    window = window.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if window.shape[1] < min_assets:
        raise ValueError(
            f"Only {window.shape[1]} assets have complete prices in the required window; "
            f"minimum is {min_assets}."
        )
    return window


def resolve_universes(
    source: str,
    *,
    local_path: str | Path | None = None,
) -> tuple[list[UniverseSpec], pd.DataFrame]:
    all_requested = unique_preserving_order(
        DJIA_CONSTITUENTS + SP100_STYLE_CANDIDATES + MULTI_ASSET_ETFS
    )
    earliest_needed = (
        pd.Timestamp(min(CONSTRUCTION_DATES)) - pd.DateOffset(months=max(ESTIMATION_WINDOWS_MONTHS)) - pd.DateOffset(days=7)
    ).strftime("%Y-%m-%d")
    latest_needed = (
        pd.Timestamp(max(CONSTRUCTION_DATES)) + pd.DateOffset(months=12) + pd.DateOffset(days=7)
    ).strftime("%Y-%m-%d")

    prices = load_price_source(
        source,
        all_requested,
        earliest_needed,
        latest_needed,
        local_path=local_path,
    )

    specs = [
        UniverseSpec("djia", "DJIA Constituents", DJIA_CONSTITUENTS),
        UniverseSpec(
            "sp100_style_top100",
            "S&P 100-style Top 100",
            SP100_STYLE_CANDIDATES,
        ),
        UniverseSpec("multi_asset_etf", "Multi-Asset ETF", MULTI_ASSET_ETFS),
    ]
    return specs, prices


def make_constructors(n_assets: int):
    n_clusters = max(2, int(math.floor(math.sqrt(n_assets))))
    return [
        ("equal_weight", EqualWeightConstructor(), {}),
        ("naive_risk_parity", NaiveRiskParity(), {}),
        ("markowitz", Markowitz(), {"ret_kind": "simple", "allow_short": False}),
        ("hrp_recursive", HRPRecursive(distance="deprado"), {}),
        (
            "hca_deprado_ew_nrp",
            HRPStyle(
                distance="deprado",
                inner=EqualWeightConstructor(),
                outer=NaiveRiskParity(),
                n_clusters=n_clusters,
                display_name="HCA De Prado EW/NRP",
            ),
            {"n_clusters": n_clusters},
        ),
    ]


def previous_available_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    candidates = index[index < date]
    if len(candidates) == 0:
        raise ValueError(f"No available date before {date.date()}.")
    return pd.Timestamp(candidates.max())


def next_available_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    candidates = index[index >= date]
    if len(candidates) == 0:
        raise ValueError(f"No available date on or after {date.date()}.")
    return pd.Timestamp(candidates.min())


def last_available_on_or_before(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    candidates = index[index <= date]
    if len(candidates) == 0:
        raise ValueError(f"No available date on or before {date.date()}.")
    return pd.Timestamp(candidates.max())


def save_mc_terminal_comparison_plot(
    run_dir: Path,
    run_id: str,
    terminal_values: pd.DataFrame,
) -> Path:
    fig = go.Figure()
    methods = list(terminal_values.columns)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]

    for idx, method in enumerate(methods):
        values = terminal_values[method].dropna().to_numpy(dtype=float)
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=method,
                histnorm="probability density",
                opacity=0.45,
                nbinsx=min(60, max(20, len(values) // 20)),
                marker=dict(color=color),
            )
        )
        fig.add_vline(
            x=float(np.mean(values)),
            line_dash="dash",
            line_color=color,
            line_width=2,
        )

    fig.update_layout(
        title=f"Monte Carlo Terminal Distribution Comparison - {run_id}",
        xaxis_title="Terminal Value",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    output_path = run_dir / "plots" / "mc_terminal_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path


def run_one_experiment(
    *,
    prices: pd.DataFrame,
    spec: UniverseSpec,
    source: str,
    construction_date: str,
    estimation_months: int,
    save_plots: bool,
) -> dict[str, object]:
    construction_ts = pd.Timestamp(construction_date)
    desired_construction_start = construction_ts - pd.DateOffset(months=estimation_months)
    desired_backtest_end = construction_ts + pd.DateOffset(months=12)

    universe_prices = available_prices(
        prices,
        spec.tickers,
        start=desired_construction_start - pd.DateOffset(days=7),
        end=desired_backtest_end + pd.DateOffset(days=7),
    )
    returns = universe_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    finite_columns = returns.columns[np.isfinite(returns.to_numpy()).all(axis=0)]
    returns = returns.loc[:, finite_columns]
    nonzero_vol_columns = returns.std(ddof=1)
    returns = returns.loc[:, nonzero_vol_columns[nonzero_vol_columns > 0].index]
    universe_prices = universe_prices.loc[:, returns.columns]
    if returns.shape[1] < 2:
        raise ValueError("Fewer than 2 usable assets remain after return cleaning.")

    if returns.index.min() > desired_construction_start + pd.DateOffset(days=7):
        raise ValueError(
            "Not enough history for the requested estimation window: "
            f"need data around {desired_construction_start.date()}, "
            f"first available return is {returns.index.min().date()}."
        )

    construction_start = next_available_date(
        returns.index,
        desired_construction_start,
    )
    construction_end = previous_available_date(returns.index, construction_ts)
    backtest_start = next_available_date(returns.index, construction_ts)
    if returns.index.max() < desired_backtest_end - pd.DateOffset(days=7):
        raise ValueError(
            "Not enough forward data for the 12-month out-of-sample backtest: "
            f"need data around {desired_backtest_end.date()}, "
            f"last available return is {returns.index.max().date()}."
        )
    backtest_end = last_available_on_or_before(returns.index, desired_backtest_end)

    if construction_start >= construction_end:
        raise ValueError("Construction window is empty after date alignment.")
    if backtest_start >= backtest_end:
        raise ValueError("Backtest window is empty after date alignment.")

    construction_returns = returns.loc[
        (returns.index >= construction_start) & (returns.index <= construction_end)
    ]
    usable_columns = construction_returns.columns[
        np.isfinite(construction_returns.to_numpy()).all(axis=0)
    ]
    construction_std = construction_returns.loc[:, usable_columns].std(ddof=1)
    usable_columns = construction_std[construction_std > 0].index
    returns = returns.loc[:, usable_columns]
    universe_prices = universe_prices.loc[:, usable_columns]
    if returns.shape[1] < 2:
        raise ValueError("Fewer than 2 usable assets remain after construction-window cleaning.")

    run_id = f"{spec.universe_id}_{construction_ts:%Y%m%d}_{estimation_months}m"
    tickers = list(universe_prices.columns)
    market_data = StandardizedData(
        prices=universe_prices,
        returns=returns,
        tickers=tickers,
        metadata={
            "source": source,
            "universe_id": spec.universe_id,
            "display_name": spec.display_name,
            "requested_tickers": spec.tickers,
            "available_tickers": tickers,
            "construction_date": str(construction_ts.date()),
            "estimation_months": estimation_months,
        },
    )

    universe = Universe(
        loader=market_data,
        tickers=tickers,
        start=str(universe_prices.index.min().date()),
        end=str(universe_prices.index.max().date()),
        construction_start=str(construction_start.date()),
        construction_end=str(construction_end.date()),
        universe_name=run_id,
        base_output_dir=FRAMEWORK_DIR,
        auto_save_data=False,
    ).prepare_data()

    constructors = make_constructors(len(tickers))
    method_names = []
    for label, constructor, kwargs in constructors:
        universe.build(constructor, label=label, set_active=False, **kwargs)
        method_names.append(label)

    backtest_results = Backtester.run_all(
        universe,
        start_date=backtest_start,
        end_date=backtest_end,
        ann_factor=ANNUALIZATION_FACTOR,
        attach=True,
    )
    mc_horizon = len(universe.get_returns_window(backtest_start, backtest_end))
    mc_results = MonteCarloEngine.run_all(
        universe,
        horizon=mc_horizon,
        n_simulations=MC_SIMULATIONS,
        initial_value=MC_INITIAL_VALUE,
        seed=MC_SEED,
        attach=True,
    )
    if save_plots:
        PortfolioVisualizer(universe).save_everything(max_mc_paths=100)
    else:
        universe.save_market_data()
        universe.save_all_constructions()
        universe.save_all_backtests()
        universe.save_all_monte_carlo()

    config = {
        "run_id": run_id,
        "source": source,
        "universe_id": spec.universe_id,
        "universe_display_name": spec.display_name,
        "requested_tickers": spec.tickers,
        "available_tickers": tickers,
        "n_assets": len(tickers),
        "construction_date": str(construction_ts.date()),
        "estimation_months": estimation_months,
        "construction_start": str(construction_start.date()),
        "construction_end": str(construction_end.date()),
        "backtest_start": str(backtest_start.date()),
        "backtest_end": str(backtest_end.date()),
        "mc_horizon": mc_horizon,
        "mc_simulations": MC_SIMULATIONS,
        "mc_seed": MC_SEED,
        "methods": method_names,
        "hca_n_clusters": max(2, int(math.floor(math.sqrt(len(tickers))))),
    }
    (universe.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    rows = []
    terminal_rows = []
    terminal_values_wide: dict[str, np.ndarray] = {}
    for name in method_names:
        bt = backtest_results[name].summary_metrics
        mc = mc_results[name].summary_metrics
        weights = universe.get_construction(name).weights
        terminal_values = np.asarray(mc_results[name].terminal_values, dtype=float)
        terminal_values_wide[name] = terminal_values
        terminal_rows.extend(
            {
                "run_id": run_id,
                "universe_id": spec.universe_id,
                "construction_date": str(construction_ts.date()),
                "estimation_months": estimation_months,
                "method": name,
                "simulation_id": sim_idx,
                "terminal_value": float(value),
            }
            for sim_idx, value in enumerate(terminal_values)
        )
        rows.append(
            {
                **config,
                "method": name,
                "total_return": bt["total_return"],
                "annualized_return": bt["annualized_return"],
                "volatility": bt["annualized_volatility"],
                "sharpe_ratio": bt["sharpe_ratio"],
                "max_drawdown": bt["max_drawdown"],
                "mean_terminal_value": mc["mean_terminal_value"],
                "median_terminal_value": mc["median_terminal_value"],
                "std_terminal_value": mc["std_terminal_value"],
                "prob_loss": mc["prob_loss"],
                "herfindahl": float((weights**2).sum()),
                "max_weight": float(weights.max()),
            }
        )
    terminal_values_frame = pd.DataFrame(terminal_values_wide)
    terminal_values_frame.to_csv(universe.output_dir / "plots" / "mc_terminal_values.csv", index=False)
    if save_plots:
        save_mc_terminal_comparison_plot(universe.output_dir, run_id, terminal_values_frame)
    return {"config": config, "rows": rows, "terminal_rows": terminal_rows}


def write_skipped(skipped: list[dict[str, str]]) -> None:
    if not skipped:
        return
    pd.DataFrame(skipped).to_csv(TABLE_DIR / "skipped_runs.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final thesis experimental setup.")
    parser.add_argument(
        "--source",
        choices=["processed", "local", "yfinance"],
        default="processed",
        help="Use the processed dataset, a local Yahoo-style CSV snapshot, or Yahoo Finance download logic.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to an existing Yahoo-style CSV snapshot. Required when --source local.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first unsupported run instead of writing skipped_runs.csv.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/JSON outputs only. Useful for quick verification runs.",
    )
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FRAMEWORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "local" and not args.local_path:
        parser.error("--local-path is required when --source local.")

    resolved_local_path = resolve_local_csv_path(args.local_path) if args.local_path else None
    specs, prices = resolve_universes(args.source, local_path=resolved_local_path)
    all_rows: list[dict[str, object]] = []
    all_terminal_rows: list[dict[str, object]] = []
    run_configs: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for spec in specs:
        for construction_date in CONSTRUCTION_DATES:
            for months in ESTIMATION_WINDOWS_MONTHS:
                try:
                    result = run_one_experiment(
                        prices=prices,
                        spec=spec,
                        source=args.source,
                        construction_date=construction_date,
                        estimation_months=months,
                        save_plots=not args.skip_plots,
                    )
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    skipped.append(
                        {
                            "universe_id": spec.universe_id,
                            "construction_date": construction_date,
                            "estimation_months": str(months),
                            "reason": str(exc),
                        }
                    )
                    continue
                run_configs.append(result["config"])
                all_rows.extend(result["rows"])
                all_terminal_rows.extend(result["terminal_rows"])

    if not all_rows:
        write_skipped(skipped)
        raise RuntimeError("No experiments completed. See tables/skipped_runs.csv for details.")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(TABLE_DIR / "final_experiment_summary.csv", index=False)

    backtest_cols = [
        "run_id",
        "universe_id",
        "construction_date",
        "estimation_months",
        "method",
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    mc_cols = [
        "run_id",
        "universe_id",
        "construction_date",
        "estimation_months",
        "method",
        "mean_terminal_value",
        "median_terminal_value",
        "std_terminal_value",
        "prob_loss",
    ]
    concentration_cols = [
        "run_id",
        "universe_id",
        "construction_date",
        "estimation_months",
        "method",
        "herfindahl",
        "max_weight",
    ]
    summary[backtest_cols].to_csv(TABLE_DIR / "backtest_summary.csv", index=False)
    summary[mc_cols].to_csv(TABLE_DIR / "mc_summary.csv", index=False)
    summary[concentration_cols].to_csv(TABLE_DIR / "concentration.csv", index=False)
    pd.DataFrame(all_terminal_rows).to_csv(TABLE_DIR / "mc_terminal_values.csv", index=False)
    write_skipped(skipped)

    config_payload = {
        "source": args.source,
        "local_path": str(resolved_local_path) if resolved_local_path is not None else None,
        "yfinance_snapshot_path": str(YF_SAVE_PATH) if args.source == "yfinance" else None,
        "yfinance_catalog_path": str(YF_CATALOG_PATH) if args.source == "yfinance" else None,
        "construction_dates": CONSTRUCTION_DATES,
        "estimation_windows_months": ESTIMATION_WINDOWS_MONTHS,
        "evaluation": "12-month forward out-of-sample backtest",
        "mc_simulations": MC_SIMULATIONS,
        "mc_seed": MC_SEED,
        "completed_runs": run_configs,
        "skipped_runs": skipped,
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    print(f"Completed {len(run_configs)} runs.")
    print(f"Wrote aggregate tables to {TABLE_DIR}.")
    if skipped:
        print(f"Skipped {len(skipped)} unsupported runs. See {TABLE_DIR / 'skipped_runs.csv'}.")


if __name__ == "__main__":
    main()
