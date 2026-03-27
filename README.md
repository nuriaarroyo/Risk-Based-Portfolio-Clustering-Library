# Portfolio Optimization & Hierarchical Risk Parity Research

This repository contains an Honors Thesis library for portfolio construction and evaluation, with a strong focus on Hierarchical Risk Parity (HRP), classical Markowitz optimization, and risk-based allocation workflows.

## What The Library Covers

- Shared market-data handling through a `Universe`
- Portfolio construction with:
  `EqualWeightConstructor`, `Markowitz`, `NaiveRiskParity`, `HRPStyle`, `HRPRecursive`
- Historical backtesting with `Backtester`
- Monte Carlo simulation with `MonteCarloEngine`
- Plot export and reporting with `PortfolioVisualizer`

## Public API

The main workflow is available from a single import surface:

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
    start="2020-01-01",
    end="2020-12-31",
).prepare_data()

universe.build(EqualWeightConstructor())
universe.build(Markowitz())
universe.build(NaiveRiskParity())
universe.build(HRPStyle())

Backtester.run_all(
    universe,
    start_date="2021-01-01",
    end_date="2021-12-31",
)

MonteCarloEngine.run_all(
    universe,
    horizon=252,
    n_simulations=500,
    seed=42,
)

visualizer = PortfolioVisualizer(universe)
visualizer.save_everything()
```

## Naming Model

Constructors now distinguish between:

- `method_id`: stable machine-friendly identifier used for default construction names and exports
- `display_name`: human-friendly label used in plots and summaries

Example:

- `method_id = "naive_risk_parity"`
- `display_name = "Naive Risk Parity (1/sigma)"`

This keeps output folders stable while preserving readable titles.

## Output Convention

By default, runs are saved under:

```text
outputs/runs/<universe_name>/
├─ data/
├─ constructions/
├─ backtests/
├─ monte_carlo/
└─ plots/
```

Default construction folders use stable names such as:

```text
outputs/runs/thesis_demo/constructions/naive_risk_parity/
outputs/runs/thesis_demo/constructions/hrp_style/
```

If you pass a custom `label=...` when building, that label becomes the saved construction name.

## Repository Structure

```text
honores_actuaria/
├─ portafolios/          # reusable library code
├─ legacy/               # archived pre-refactor code
├─ notebooks/
│  ├─ exploration/       # scratch work and prototyping
│  ├─ demos/             # polished walkthroughs and examples
│  └─ validation/        # notebook-based checks
├─ data/
│  └─ processed/         # cleaned local datasets
├─ outputs/
│  ├─ runs/              # main exported runs by universe/demo
│  ├─ test_runs/         # generated outputs from test/demo runs
│  ├─ plots/             # loose exported plots
│  └─ data_exports/      # generated csv/html artifacts
└─ README.md
```

## Notes

- The library is now English-first at the public API level.
- Legacy code stays isolated under `legacy/`.
- Existing Spanish method names remain as compatibility aliases while the English API becomes the main path.
