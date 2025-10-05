# log-anomaly-detection-mvp
Quickstart:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python -m src  # sanity check

## Generate sample log data with provisioned configuration 
### Data generated with anomaly
python -m log_anomaly_detection.data.loader --n 10000 --out data/sample_logs.parquet --anomaly burst --frac 0.02 --seed 7

### Data generated without anomaly
python -m log_anomaly_detection.data.loader --n 10000 --out data/sample_logs.parquet --anomaly none

## EDA Notebook: `notebooks/01_eda.ipynb`
**Why:** Establish a baseline for "normal" log behavior before modeling anomalies.  
This notebook helps you (a) see latency shape/tails, (b) read p95/p99 directly,  
(c) understand status-code mix and failure rate, and (d) spot hourly traffic patterns.

**Input data:** `data/sample_logs.parquet`  
Generate it if missing

