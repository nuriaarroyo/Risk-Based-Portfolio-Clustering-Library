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

This mode can write a raw snapshot file and depends on network access and current Yahoo Finance availability.

## Where To Read Results

- Curated thesis-facing outputs live in `outputs/data_exports/final_experimental_setup/`.
- Thesis notebooks live in `notebooks/thesis/`.
- The active results-reading workbench is `notebooks/final_experimental_setup/read_final_results.ipynb`.

## Notes

- Large generated outputs are intentionally not the main tracked face of the repository.
- The cleanup branch keeps the project presentation focused on code, thesis notebooks, and curated final exports.
