# Bike Share Demand Forecast

A chronological one-step-ahead hourly demand forecast using lagged demand, calendar and weather features.

## Dataset

[UCI Bike Sharing dataset (ID 275)](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset), with 17,379 hourly observations. The project uses the hourly `hour.csv` table and target `cnt`. UCI dataset citation: Fanaee-T (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894). The dataset describes Washington, D.C. bike-share usage in 2011–2012.

## Run

```bash
python -m pip install -r requirements.txt
python -c "from ucimlrepo import fetch_ucirepo; import pandas as pd; d=fetch_ucirepo(id=275); pd.concat([d.data.features,d.data.targets],axis=1).to_csv('hour.csv',index=False)"
python main.py --input hour.csv --out outputs
```

The runner saves local metrics, chronological forecasts, and a serialized estimator. Data and outputs are Git-ignored.

## Holdout results

The last 20% of time-ordered rows are the holdout (3,471 hours); evaluation is one-step-ahead, using previously observed counts for lag features.

| Model | MAE (rides/hour) | RMSE (rides/hour) |
|---|---:|---:|
| HistGradientBoosting with lag/calendar/weather features | 38.81 | 58.78 |
| Previous-hour count (lag-1 baseline) | 146.48 | — |
| Previous-day count (lag-24 baseline) | 168.57 | — |

This is a single forward split, not a multi-day forecast experiment. The historical two-year dataset does not establish performance for present-day demand, new stations, or a different city. Validate future forecast horizons and station-level operational constraints before planning rebalancing.

Full aggregate metrics are in [`reports/benchmark.json`](reports/benchmark.json).
