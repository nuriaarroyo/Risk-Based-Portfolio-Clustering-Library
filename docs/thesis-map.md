# Thesis Map

This document is the shortest path through the repository for a thesis reader.

## Start Here

- [README.md](../README.md) gives the high-level project overview and repository layout.
- [notebooks/README.md](../notebooks/README.md) explains how notebooks are split between thesis material, active work, and archive.
- [outputs/data_exports/final_experimental_setup/README.md](../outputs/data_exports/final_experimental_setup/README.md) explains the canonical thesis-facing export surface.

## Core Code

- `portafolios/` contains the active library code.
- `scripts/run_final_experimental_setup.py` is the main thesis experiment pipeline.
- `legacy/` contains archived pre-refactor code kept only for reference.

## Thesis-Facing Notebooks

- `notebooks/thesis/final_experimental_setup/` contains the paper-facing visualization notebooks.
- `notebooks/thesis/thesis_minimal_results/` contains the compact thesis results notebook.

## Active Workbench

- `notebooks/final_experimental_setup/read_final_results.ipynb` is the live notebook for ongoing analysis and final checks.

## Archived Reference Material

- `notebooks/archive/demos/` contains architecture and library walkthrough notebooks.
- `notebooks/archive/validation/` contains notebook-based checks and validation material.
- `notebooks/archive/exploration/` contains scratch exploration.
- `notebooks/archive/final_experimental_setup_assets/` contains exported images that are not part of the live notebook workspace.

## Final Thesis Outputs

- `outputs/data_exports/final_experimental_setup/paper_figures/` contains curated thesis and presentation figures.
- `outputs/data_exports/final_experimental_setup/tables/` contains aggregated summary tables.
- `outputs/data_exports/final_experimental_setup/experiment_config.json` records the top-level configuration for that export set.

## Generated Run Artifacts

- The library writes per-run artifacts such as `data/`, `constructions/`, `backtests/`, `monte_carlo/`, and `plots/`.
- These large generated outputs are intentionally treated as reproducible local artifacts rather than the main tracked thesis surface.
