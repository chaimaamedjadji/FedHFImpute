# prepare_beijingpm25_fedhf.py
# ------------------------------------------------------------
# UCI Beijing PM2.5 dataset (PRSA_data_2010.1.1-2014.12.31.csv)
# Missing values are denoted as "NA". :contentReference[oaicite:5]{index=5}
#
# Creates FL clients with heterogeneous schemas + global corr graph.
# ------------------------------------------------------------

from __future__ import annotations
import os, argparse, urllib.request
import numpy as np
import pandas as pd


URL_CSV = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"


def download_if_missing(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print("[DOWNLOAD]", url)
    urllib.request.urlretrieve(url, path)


def zscore_train(Xtr: np.ndarray, obs_tr: np.ndarray):
    F = Xtr.shape[1]
    mu = np.zeros(F, np.float32)
    sd = np.ones(F, np.float32)
    for f in range(F):
        obs = obs_tr[:, f]
        if obs.sum() < 5:
            continue
        col = Xtr[obs, f]
        mu[f] = col.mean()
        sd[f] = col.std() + 1e-6
    return ((Xtr - mu[None, :]) / sd[None, :]).astype(np.float32), mu, sd


def build_corr_graph(Xtr: np.ndarray, obs_tr: np.ndarray, k: int = 8, min_abs_corr: float = 0.1):
    Xf = Xtr.copy()
    F = Xf.shape[1]
    for f in range(F):
        obs = obs_tr[:, f]
        if obs.sum() < 5:
            Xf[:, f] = 0.0
        else:
            m = Xf[obs, f].mean()
            Xf[~obs, f] = m

    C = np.corrcoef(Xf, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)

    edges, weights = [], []
    for i in range(F):
        idx = np.argsort(-np.abs(C[i]))[:k]
        for j in idx:
            w = float(abs(C[i, j]))
            if w < min_abs_corr:
                continue
            edges.append((i, j))
            weights.append(w)

    edge_index = np.array(edges, np.int64).T if edges else np.zeros((2, 0), np.int64)
    edge_weight = np.array(weights, np.float32) if weights else np.zeros((0,), np.float32)
    if edge_weight.size > 0:
        edge_weight /= edge_weight.max()
    return edge_index, edge_weight


def make_clients_time_split(Xtr: np.ndarray, n_clients: int, keep_ratio: float, seed: int):
    """
    For time series: split contiguous chunks as clients (more realistic than random split).
    """
    rng = np.random.default_rng(seed)
    N, F = Xtr.shape
    splits = np.array_split(np.arange(N), n_clients)

    out = []
    for cid, idx in enumerate(splits):
        Xi = Xtr[idx].copy()
        avail = np.zeros(F, np.float32)
        keep = rng.choice(F, size=max(1, int(F * keep_ratio)), replace=False)
        avail[keep] = 1.0
        Xi[:, avail == 0] = np.nan
        out.append((cid, Xi.astype(np.float32), avail.astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="beijingpm25_fedhf")
    ap.add_argument("--raw-dir", type=str, default="raw_beijingpm25")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--keep-ratio", type=float, default=0.7)
    ap.add_argument("--graph-k", type=int, default=8)
    ap.add_argument("--min-abs-corr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "clients"), exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    csv_path = os.path.join(args.raw_dir, "beijing_pm25.csv")
    if not args.offline:
        download_if_missing(URL_CSV, csv_path)
    if not os.path.exists(csv_path):
        raise RuntimeError("Missing csv. Download manually then run --offline.")

    df = pd.read_csv(csv_path, na_values=["NA", "NaN", ""])
    # Drop id/time columns (keep meteorology)
    drop_cols = [c for c in ["No", "year", "month", "day", "hour", "cbwd"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    X = df.to_numpy(dtype=np.float32)

    # Train/test split by time (first 80% train, last 20% test)
    N = X.shape[0]
    split = int(0.8 * N)
    Xtr, Xte = X[:split], X[split:]
    obs_tr = ~np.isnan(Xtr)
    obs_te = ~np.isnan(Xte)

    # Standardize using train
    Xtr_std, mu, sd = zscore_train(Xtr, obs_tr)
    Xte_std = ((Xte - mu[None, :]) / sd[None, :]).astype(np.float32)

    # Graph
    edge_index, edge_weight = build_corr_graph(Xtr_std, obs_tr, k=args.graph_k, min_abs_corr=args.min_abs_corr)

    np.savez_compressed(
        os.path.join(args.out_dir, "global_graph.npz"),
        edge_index=edge_index,
        edge_weight=edge_weight,
        mu=mu,
        sd=sd,
    )

    # Clients = contiguous time chunks
    clients = make_clients_time_split(Xtr_std, args.n_clients, args.keep_ratio, args.seed)
    for cid, Xi, avail in clients:
        np.savez_compressed(
            os.path.join(args.out_dir, "clients", f"client_{cid}.npz"),
            X=Xi,
            availability_mask=avail,
        )

    np.savez_compressed(os.path.join(args.out_dir, "test.npz"), X=Xte_std, obs=obs_te)
    print(f"[OK] BeijingPM2.5 prepared at {args.out_dir} with F={X.shape[1]}")


if __name__ == "__main__":
    main()
