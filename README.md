# Portfolio Optimization Research Library

Modular Python library developed for an Honors Thesis in Actuarial Science, focused on comparative portfolio construction with a strong emphasis on Hierarchical Risk Parity (HRP), Markowitz optimization, and risk-based allocation workflows.

## Why This Project

This project was built to move beyond notebook-only experimentation into a reusable research framework. It supports a full workflow for:

- loading and standardizing market data
- constructing portfolios with multiple allocation methods
- separating in-sample construction from out-of-sample backtesting
- running Monte Carlo simulations
- exporting plots and structured results for analysis

## Core Features

- Shared `Universe` object for data, constructions, metadata, and exports
- Portfolio constructors:
  `EqualWeightConstructor`, `Markowitz`, `NaiveRiskParity`, `HRPStyle`, `HRPRecursive`
- Out-of-sample backtesting with `Backtester`
- Monte Carlo simulation with `MonteCarloEngine`
- Plotly-based reporting with `PortfolioVisualizer`
- Reproducible notebook demos and validation workflows

## Public API

The main workflow is exposed from a single import surface:

```python
from portafolios import (
    Universe,
    local_loader,
    EqualWeightConstructor,
    Markowitz,
    NaiveRiskParity,
    HRPStyle,
    HRPRecursive,
    Backtester,
    MonteCarloEngine,
    PortfolioVisualizer,
)
```

## Quick Example

```python
from pathlib import Path

from portafolios import (
    Universe,
    local_loader,
    EqualWeightConstructor,
    Markowitz,
    NaiveRiskParity,
    HRPStyle,
    Backtester,
    MonteCarloEngine,
    PortfolioVisualizer,
)

project_root = Path.cwd()
csv_path = project_root / "data" / "yf_snapshot.csv"

universe = Universe(
    universe_name="thesis_demo",
    loader=local_loader,
    loader_kwargs={"path": csv_path, "prefer_adj_close": True},
    tickers=["AAPL", "MSFT", "AMZN", "GOOG"],
    start="2024-01-02",
    end="2024-03-28",
    construction_start="2024-01-02",
    construction_end="2024-02-15",
    auto_save_data=False,
).prepare_data()

universe.build(EqualWeightConstructor())
universe.build(Markowitz(), ret_kind="simple", allow_short=False)
universe.build(NaiveRiskParity())
universe.build(HRPStyle())

Backtester.run_all(universe)
MonteCarloEngine.run_all(universe, horizon=252, n_simulations=500, seed=42)

visualizer = PortfolioVisualizer(universe)
visualizer.save_everything()
```

## Workflow Model

The library supports a two-stage research workflow:

- `start` / `end` define the full market-data horizon loaded into the universe
- `construction_start` / `construction_end` define the shared in-sample training window for all constructors
- `Backtester.run()` defaults to the first available date after `construction_end` through the last available return in the universe

This keeps portfolio comparisons fair while preserving a clean out-of-sample evaluation path.

You can also inspect any subwindow of the backtest series:

```python
bt = Backtester(universe, "hrp_style")
result = bt.run()

window_summary = bt.summarize_window(
    result,
    start_date="2021-03-01",
    end_date="2021-06-30",
)

fig = PortfolioVisualizer(universe).plot_backtest(
    "hrp_style",
    start_date="2021-03-01",
    end_date="2021-06-30",
)
```

## Output Structure

The library writes a standard per-run structure under:

```text
outputs/runs/<universe_name>/
|- data/
|- constructions/
|- backtests/
|- monte_carlo/
`- plots/
   |- constructions/
   |- backtests/
   |- monte_carlo/
   `- comparisons/
```

The storage split is intentional:

- `constructions/`, `backtests/`, and `monte_carlo/` keep saved weights, return series, simulation tables, and JSON metadata
- `plots/` keeps HTML figures only, grouped by plot type instead of mixing them into the data folders
- cross-method Monte Carlo comparison data is stored at `monte_carlo/comparison_terminal_values.csv`

Constructors use:

- `method_id` for stable folder-safe identifiers such as `naive_risk_parity`
- `display_name` for human-readable plot titles and summaries

For the thesis workflow, the experiment pipeline also maintains a curated export surface under:

```text
outputs/data_exports/final_experimental_setup/
|- paper_figures/
|- tables/
`- experiment_config.json
```

When the thesis runner uses `--source yfinance`, it reuses the saved snapshot at
`outputs/final_experimental_setup/data/yfinance_final_setup_raw.csv` if it already
exists; otherwise it downloads that file and records it in
`outputs/final_experimental_setup/data/yfinance_snapshot_catalog.json`.

Large generated run artifacts are treated as reproducible local outputs rather than the main thesis-facing repository surface.

## Documentation

- [docs/thesis-map.md](docs/thesis-map.md) gives a quick guide to where thesis-facing code, notebooks, and outputs live.
- [docs/reproducibility.md](docs/reproducibility.md) lists the main commands for rerunning the thesis pipeline.
- [notebooks/README.md](notebooks/README.md) explains the notebook layout in more detail.

## Repository Structure

```text
honores_actuaria/
|- docs/               # thesis map and reproducibility notes
|- portafolios/        # active library code
|- scripts/            # runnable experiment pipelines
|- legacy/             # archived pre-refactor code
|- notebooks/
|  |- thesis/          # final thesis-facing notebooks
|  |- final_experimental_setup/
|  |  `- read_final_results.ipynb   # active working notebook
|  `- archive/         # demos, validation, exploration, loose exports
|- data/
|  |- yf_snapshot.csv
|  `- processed/
|- outputs/
|  `- data_exports/
|     `- final_experimental_setup/
|        |- paper_figures/
|        `- tables/
`- README.md
```

## Current Status

This is an active thesis project, not a finished production package. The core research workflow is implemented and runnable, and the repository includes a working library structure, curated thesis notebooks, and reproducible exported results.

The current cleanup has focused on repository presentation rather than changing the core library logic:

- thesis-facing notebooks were separated from demos, validation, and exploration material
- loose exported images were moved out of the live notebook workspace
- large generated outputs were removed from git tracking while remaining reproducible locally

Packaging and final polish are still in progress.
