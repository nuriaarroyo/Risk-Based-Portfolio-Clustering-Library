# Portfolio Optimization & Hierarchical Risk Parity Research

This repository contains an Honors Thesis library for Actuarial Science focused on portfolio construction, especially HRP and related risk-based allocation methods.

## Project Structure

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

## Storage Rules

- Put reusable Python code in `portafolios/`.
- Keep old implementations in `legacy/`.
- Store notebooks only in `notebooks/`.
- Save generated artifacts in `outputs/`, never at the repository root.
- Keep input datasets in `data/`, and cleaned local files in `data/processed/`.

## Output Convention

`PortfolioUniverse` now defaults to:

```text
outputs/runs/<universe_name>/
├─ data/
├─ constructions/
├─ backtests/
├─ monte_carlo/
└─ plots/
```

This keeps experiments reproducible while preventing root-level clutter.
