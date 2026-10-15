# Bike Share Demand Forecast

A chronological gradient-boosting demand forecast with lag features, MAE/RMSE, holdout forecasts, and a serialized estimator.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A chronological gradient-boosting demand forecast with lag features, MAE/RMSE, holdout forecasts, and a serialized estimator. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). UCI `hour.csv` or compatible CSV with `dteday` and `cnt` columns in chronological order. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/hour.csv --date dteday --target cnt --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

Historical demand is not a guarantee of future demand. The first lag features assume rows are a regular time grid; remove gaps or resample explicitly before relying on them.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

Code: MIT. Third-party datasets remain subject to their original terms.
