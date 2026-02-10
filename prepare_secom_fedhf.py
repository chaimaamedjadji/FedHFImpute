# prepare_secom_fedhf.py
# ---------------------------------------------------------
# Robust SECOM preparation (no ucimlrepo), with QUALITY FILTERS:
#  - Downloads secom.data (and optionally secom.labels) unless --offline
#  - Drops near-constant features (variance filter)
#  - Optionally drops very-missing features (missing-rate filter)
#  - Train/test split (avoid leakage in correlation graph)
#  - Standardize per-feature using TRAIN observed values only
#  - Build GLOBAL weighted correlation graph on TRAIN
#  - Create FL clients with heterogeneous schemas (availability_mask)
#  - Saves mappings for kept features
#
# Outputs (out_dir):
#   global_graph.npz: edge_index, edge_weight, mu, sd, kept_feature_idx
#   clients/client_{i}.npz: X, availability_mask
#   test.npz: X, obs
#
# Usage:
#   python prepare_secom_fedhf.py
#   python prepare_secom_fedhf.py --offline --raw-dir raw_secom
#   python prepare_secom_fedhf.py --offline --raw-dir raw_secom --keep-ratio 0.8 --graph-k 8
# ---------------------------------------------------------

from __future__ import annotations
import os
import argparse
import urllib.request
import numpy as np


UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
UCI_DATA_URL = UCI_BASE + "secom.data"
UCI_LABELS_URL = UCI_BASE + "secom.labels"


def download_if_missing(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    print(f"[DOWNLOAD] {url}")
    urllib.request.urlretrieve(url, dst_path)
    if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        raise RuntimeError(f"Download failed or empty file: {dst_path}")


def load_secom_from_files(data_path: str, labels_path: str | None):
    """
    secom.data: space-separated floats, missing values as 'NaN'
    secom.labels: label + timestamp (optional); label is +1 or -1
    """
    X = np.genfromtxt(data_path, dtype=np.float32)
    if X.ndim != 2:
        raise RuntimeError(f"Failed to parse {data_path} into 2D array. Got shape: {X.shape}")

    y = None
    if labels_path is not None and os.path.exists(labels_path):
        lab = np.genfromtxt(labels_path, dtype=str)
        if lab.ndim == 1:
            y_raw = lab
        else:
            y_raw = lab[:, 0]
        y = np.where(y_raw.astype(np.int32) == -1, 1, 0).astype(np.int64)

    return X, y


def variance_filter(X: np.ndarray, min_var: float = 1e-6, min_obs: int = 10):
    """Keep features whose observed variance >= min_var."""
    F = X.shape[1]
    var = np.zeros(F, dtype=np.float32)
    obs_count = np.zeros(F, dtype=np.int32)
    for f in range(F):
        col = X[:, f]
        m = ~np.isnan(col)
        obs_count[f] = int(m.sum())
        if obs_count[f] < min_obs:
            var[f] = 0.0
        else:
            var[f] = float(np.var(col[m]))

    keep = var >= float(min_var)
    return keep, var, obs_count


def missing_rate_filter(X: np.ndarray, max_missing: float = 0.95):
    """Keep features whose missing rate <= max_missing."""
    miss_rate = np.mean(np.isnan(X), axis=0).astype(np.float32)
    keep = miss_rate <= float(max_missing)
    return keep, miss_rate


def zscore_per_feature_train(X_train: np.ndarray, obs_train: np.ndarray):
    """Compute mean/std per feature from TRAIN observed entries only."""
    F = X_train.shape[1]
    mu = np.zeros(F, dtype=np.float32)
    sd = np.ones(F, dtype=np.float32)

    for f in range(F):
        obs = obs_train[:, f]
        if obs.sum() < 5:
            mu[f] = 0.0
            sd[f] = 1.0
            continue
        col = X_train[obs, f]
        mu[f] = float(col.mean())
        sd[f] = float(col.std()) + 1e-6

    X_train_std = (X_train - mu[None, :]) / sd[None, :]
    return X_train_std.astype(np.float32), mu, sd


def build_global_corr_graph(X_train_std: np.ndarray, obs_train: np.ndarray, k: int = 8):
    """
    Weighted directed edges from abs correlation on TRAIN.
    Missing values filled with feature mean for corr computation.
    """
    Xf = X_train_std.copy()
    F = Xf.shape[1]

    for f in range(F):
        obs = obs_train[:, f]
        if obs.sum() < 5:
            Xf[:, f] = 0.0
        else:
            mu = float(Xf[obs, f].mean())
            Xf[~obs, f] = mu

    C = np.corrcoef(Xf, rowvar=False)  # (F, F)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)

    edges = []
    weights = []
    for i in range(F):
        # top-k by abs corr
        idx = np.argsort(-np.abs(C[i]))[:k]
        for j in idx:
            if i == j:
                continue
            w = float(abs(C[i, j]))
            if w <= 0.0:
                continue
            edges.append((i, j))
            weights.append(w)

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32) if weights else np.zeros((0,), dtype=np.float32)

    # normalize to [0,1]
    if edge_weight.size > 0:
        m = float(edge_weight.max())
        if m > 0:
            edge_weight = edge_weight / m

    return edge_index, edge_weight


def make_clients(X_train_std: np.ndarray, n_clients: int, keep_ratio: float, missing_extra: float, seed: int):
    """
    Create heterogeneous-schema clients from TRAIN:
      - split rows across clients
      - each client has random subset of available features
      - unavailable features set to NaN
      - optional extra missingness on available features
    """
    rng = np.random.default_rng(seed)
    N, F = X_train_std.shape

    perm = rng.permutation(N)
    splits = np.array_split(perm, n_clients)

    clients = []
    for cid, idx in enumerate(splits):
        Xi = X_train_std[idx].copy()

        avail = np.zeros(F, dtype=np.float32)
        keep = rng.choice(F, size=max(1, int(F * keep_ratio)), replace=False)
        avail[keep] = 1.0

        Xi[:, avail == 0] = np.nan

        if missing_extra > 0:
            miss = rng.uniform(size=Xi.shape) < missing_extra
            miss[:, avail == 0] = True
            Xi[miss] = np.nan

        clients.append((cid, Xi.astype(np.float32), avail.astype(np.float32)))

    return clients


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="secom_fedhf")
    p.add_argument("--raw-dir", type=str, default="raw_secom")
    p.add_argument("--offline", action="store_true", help="Do not download; expect raw files already present.")

    p.add_argument("--n-clients", type=int, default=5)
    p.add_argument("--keep-ratio", type=float, default=0.6)
    p.add_argument("--missing-extra", type=float, default=0.0)

    p.add_argument("--graph-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # Quality filters
    p.add_argument("--min-var", type=float, default=1e-6, help="Variance threshold on observed entries.")
    p.add_argument("--min-obs", type=int, default=10, help="Min observed count needed to compute variance.")
    p.add_argument("--max-missing", type=float, default=0.95, help="Drop features with missing rate > this.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "clients"), exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    data_path = os.path.join(args.raw_dir, "secom.data")
    labels_path = os.path.join(args.raw_dir, "secom.labels")

    if not args.offline:
        download_if_missing(UCI_DATA_URL, data_path)
        # labels are optional; attempt download but don't fail if blocked
        try:
            download_if_missing(UCI_LABELS_URL, labels_path)
        except Exception as ex:
            print(f"[WARN] Could not download labels ({ex}). Continuing without labels.")
            labels_path = None
    else:
        if not os.path.exists(data_path):
            raise RuntimeError(f"Offline mode: missing {data_path}")
        if not os.path.exists(labels_path):
            print(f"[WARN] Offline mode: missing {labels_path} (labels will be skipped)")
            labels_path = None

    X, y = load_secom_from_files(data_path, labels_path)
    N0, F0 = X.shape
    print(f"[INFO] Loaded raw SECOM: N={N0}, F={F0}")

    # --- Quality filters ---
    keep_var, var, obs_count = variance_filter(X, min_var=args.min_var, min_obs=args.min_obs)
    keep_miss, miss_rate = missing_rate_filter(X, max_missing=args.max_missing)
    keep = keep_var & keep_miss
    kept_feature_idx = np.where(keep)[0].astype(np.int64)

    X = X[:, keep]
    N, F = X.shape
    obs = ~np.isnan(X)

    print(f"[INFO] Variance filter kept: {int(keep_var.sum())}/{F0}")
    print(f"[INFO] Missing-rate filter kept: {int(keep_miss.sum())}/{F0}")
    print(f"[INFO] Combined kept: {F}/{F0}")

    # --- Train/test split ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train = X[train_idx]
    obs_train = obs[train_idx]

    X_test = X[test_idx]
    obs_test = obs[test_idx]

    # --- Standardize using TRAIN observed stats ---
    X_train_std, mu, sd = zscore_per_feature_train(X_train, obs_train)
    X_test_std = ((X_test - mu[None, :]) / sd[None, :]).astype(np.float32)

    # --- Build global correlation graph on TRAIN ---
    edge_index, edge_weight = build_global_corr_graph(X_train_std, obs_train, k=args.graph_k)

    np.savez_compressed(
        os.path.join(args.out_dir, "global_graph.npz"),
        edge_index=edge_index,
        edge_weight=edge_weight,
        mu=mu,
        sd=sd,
        kept_feature_idx=kept_feature_idx,
    )

    # --- Create clients from TRAIN only ---
    clients = make_clients(
        X_train_std,
        n_clients=args.n_clients,
        keep_ratio=args.keep_ratio,
        missing_extra=args.missing_extra,
        seed=args.seed,
    )
    for cid, Xi, avail in clients:
        np.savez_compressed(
            os.path.join(args.out_dir, "clients", f"client_{cid}.npz"),
            X=Xi,
            availability_mask=avail,
        )

    np.savez_compressed(
        os.path.join(args.out_dir, "test.npz"),
        X=X_test_std,
        obs=obs_test,
        y=y[test_idx] if y is not None else None,
        kept_feature_idx=kept_feature_idx,
    )

    print(f"[OK] Prepared SECOM FedHF dataset at: {args.out_dir}")
    print(f"  After filtering: N={N}, F={F}")
    print(f"  Clients: {args.n_clients}, keep_ratio={args.keep_ratio}, missing_extra={args.missing_extra}")
    print(f"  Global graph edges: {edge_index.shape[1]} (top-k={args.graph_k})")
    print(f"  Outputs: global_graph.npz, clients/client_*.npz, test.npz")


if __name__ == "__main__":
    main()
