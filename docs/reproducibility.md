# Reproducibility

This project currently assumes an existing Python environment with the required scientific Python stack already installed.


## Main Thesis Pipeline

From the repository root, run:

```powershell
python scripts/run_final_experimental_setup.py --source processed
```

This writes the experiment outputs under:

```text
outputs/final_experimental_setup/
```

## Existing Local Snapshot

If you already have a Yahoo-style CSV snapshot, you can run the full experiment through the local loader instead of the download-oriented path:

```powershell
python scripts/run_final_experimental_setup.py --source local --local-path outputs/final_experimental_setup/data/yfinance_final_setup_raw.csv
```

This is the cleanest way to reuse an existing snapshot without relying on live download behavior.

## Faster Verification Run

For a lighter run that skips Plotly HTML generation:

```powershell
python scripts/run_final_experimental_setup.py --source processed --skip-plots
```

This is useful for verifying tables and core pipeline logic without regenerating the full plot set.

## Optional Yahoo Finance Mode

If you want to use downloaded data instead of the processed local dataset:

```powershell
python scripts/run_final_experimental_setup.py --source yfinance
```

This mode now follows a simple reuse-first rule:

- if `outputs/final_experimental_setup/data/yfinance_final_setup_raw.csv` already exists, the runner reuses it through `local_loader`
- if it does not exist, the runner downloads it with `yfinance`, saves it to that exact path, and registers it in `outputs/final_experimental_setup/data/yfinance_snapshot_catalog.json`

That keeps `--source yfinance` as the main entrypoint while still making the saved snapshot reusable as a local CSV source.

## Where To Read Results

- Curated thesis-facing outputs live in `outputs/data_exports/final_experimental_setup/`.
- Thesis notebooks live in `notebooks/thesis/`.
- The active results-reading workbench is `notebooks/final_experimental_setup/read_final_results.ipynb`.

## Notes

- Large generated outputs are intentionally not the main tracked face of the repository.
- The cleanup branch keeps the project presentation focused on code, thesis notebooks, and curated final exports.
