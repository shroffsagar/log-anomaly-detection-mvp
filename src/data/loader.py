import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def make_msg(response_code: int, endpoint: str):
    if response_code >= 500:
        return f"server error {response_code} on {endpoint}"
    elif response_code >= 400:
        return f"client error {response_code} on {endpoint}"
    else:
        return f"completed {endpoint} with {response_code}"

def generate_logs(n: int, rng:np.random.Generator) -> pd.DataFrame:
    # log timestamp
    timestamps = pd.date_range(end=pd.Timestamp.now().floor('s'), periods=n, freq="s")
    # log level
    log_levels = rng.choice(["INFO", "WARN", "ERROR"], size=n, p=[0.90, 0.08, 0.02])
    # response code
    status_codes = [200, 201, 204, 301, 302, 400, 401, 403, 404, 500, 502]
    status_p = [0.55, 0.03, 0.05, 0.03, 0.02, 0.07, 0.02, 0.01, 0.19, 0.02, 0.01]
    response_codes = rng.choice(status_codes, size=n, p=status_p)
    # endpoints
    paths = ["/", "/login", "/logout", "/api/v1/users", "/api/v1/users/123",
             "/api/v1/orders", "/api/v1/cart", "/search", "/static/app.js"]
    path_p = [0.15, 0.08, 0.02, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15]
    request_endpoints = rng.choice(paths, size=n, p=path_p)
    # response latency ms
    latencies = rng.lognormal(mean=4.6, sigma=0.5, size=n)
    latencies = np.clip(latencies, 5, 3000).round().astype(int)
    # session user ids 
    user_ids_issuing_reqs = rng.integers(1, 1001, size=n)
    # log messages
    log_messages = [make_msg(response_codes[log_line], request_endpoints[log_line]) for log_line in range(n)]
    return pd.DataFrame({
        "ts" : timestamps,
        "level": log_levels,
        "status": response_codes,
        "path": request_endpoints,
        "latency_ms": latencies,
        "user_id": user_ids_issuing_reqs,
        "msg": log_messages
    })

def inject_latency_bursts(data: pd.DataFrame, random: np.random.Generator, window_ratio: float):
    data = data.copy()
    n = len(data)
    anomaly_window_size = max(10, int(n * window_ratio))
    start:int = int(random.integers(low = 0, high = n-anomaly_window_size))
    end:int = start+anomaly_window_size
    idxs = np.arange(start, end)

    # boost latency in the data
    bump = random.uniform(6.0, 10.0)
    boosted = (data.loc[idxs, "latency_ms"] * bump).clip(50, 5000).round().astype(int)
    data.loc[idxs, 'latency_ms'] = boosted

    # flip some to 5xx + ERROR
    flip_random = random.random(anomaly_window_size) < 0.60
    flip_idxs = idxs[flip_random]
    data.loc[flip_idxs, 'level'] = 'ERROR'
    data.loc[flip_idxs, 'status'] = random.choice([500, 502], size=len(flip_idxs))
    
    # update messages
    data.loc[flip_idxs, 'msg'] = [make_msg(response_code=data.loc[i, 'status'], endpoint=data.loc[i, 'path']) for i in flip_idxs]

    return data, (start, end)

def write_parquet(df: pd.DataFrame, destination: str) -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, engine="pyarrow", index=False)

def cli():
    # define arguments
    parser = argparse.ArgumentParser(description="Generate sample logs")
    parser.add_argument("--n", type=int, default=1000, help="number of rows")
    parser.add_argument("--anomaly", choices=["burst", "none"], default="burst",
                        help="include a burst anomaly (default) or 'none' for normal-only logs")
    parser.add_argument("--frac", type=float, default=0.01,
                        help="fraction of rows in the anomaly window (default 1%)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")
    parser.add_argument("--out", type=str, default="data/sample_data.parquet", help="target file path for generated parquet file")

    args = parser.parse_args()
    print(f"Generating {args.n} rows in {args.out}")
    # generate logs
    rng = np.random.default_rng(args.seed)
    logs = generate_logs(args.n, rng)
    # induce anomalies
    if args.anomaly == "none":
        print("No anomalies injected")
    else:
        logs, (start, end) = inject_latency_bursts(logs, rng, args.frac)
        print(f"Injected anomaly window rows [{start}:{end}) length={end-start}")
        print(f"Share of 5xx overall: {(logs['status'] >= 500).mean():.2%}")
    # write to file
    write_parquet(logs, args.out)

if __name__ == "__main__":
    cli()