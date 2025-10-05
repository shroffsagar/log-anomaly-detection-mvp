from log_anomaly_detection.model.rcf_detector import RCFDetector
import argparse
import numpy as np
import pandas as pd

def load_from_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df.to_numpy()

def main():
    # X = make_synthetic_X(rows=300, cols=3, seed=seed)
    # det = RCFDetector().fit(X, n_trees=args.n_trees, tree_size=args.tree_size, random_state=args.seed)
    # det.save(args.out)
    # print(f"Saved RCFDetector to {args.out} (X={X.shape}, n_trees={...}, tree_size={...})")
    args_parser = argparse.ArgumentParser(description="Trains the model for the normal logs")
    args_parser.add_argument("--parquet_train_data_path", type=str, help="Path to training data in parquet file format")
    args = args_parser.parse_args()
    X = load_from_parquet(args.parquet_train_data_path)
    fit_model = RCFDetector().fit(X)

if __name__ == "__main__":
    main()