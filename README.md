# log-anomaly-detection-mvp
Quickstart:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python -m src  # sanity check

## Generate sample log data with provisioned configuration 
### Data generated with anomaly
python -m src.data.loader --n 10000 --out data/sample_logs.parquet --anomaly burst --frac 0.02 --seed 7

### Data generated without anomaly
python -m src.data.loader --n 10000 --out data/sample_logs.parquet --anomaly none