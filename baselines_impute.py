# baselines_impute.py (FIXED)
# ------------------------------------------------------------
# Baselines for heterogeneous-schema imputation.
# Handles columns that are entirely missing in a client split (sklearn drops them).
# Always returns predictions with the original shape (N,F).
#
# Saves per-client baseline RMSE to CSV.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import glob
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def deterministic_mask(obs: np.ndarray, avail: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    eligible = obs & avail[None, :]
    r = rng.rand(*eligible.shape)
    return eligible & (r < p)


def rmse_on_mask(x_true: np.ndarray, x_pred: np.ndarray, mask: np.ndarray) -> float:
    # Evaluate only where mask is True (guaranteed observed in x_true by construction)
    diff = (x_pred - x_true)
    se = (diff * diff)[mask].sum()
    n = mask.sum()
    return float(np.sqrt(se / max(int(n), 1)))


def load_client_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    avail = d["availability_mask"].astype(np.float32) > 0.5
    return X, avail


def make_imputer(name: str):
    if name == "mean":
        return SimpleImputer(strategy="mean")
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        return KNNImputer(n_neighbors=5, weights="distance")
    if name == "mice":
        return IterativeImputer(
            random_state=0,
            max_iter=10,
            sample_posterior=False,
            initial_strategy="mean",
        )
    raise ValueError(f"Unknown baseline: {name}")


def fit_transform_safe(imputer, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    sklearn imputers can drop all-missing columns.
    We handle this by:
      - selecting columns that have at least 1 observed value in TRAIN
      - fit+transform only those columns
      - fill other columns with 0.0 (mean in standardized space) to keep shape (N,F)
    """
    F = X_train.shape[1]

    # valid cols = not all-missing in train
    valid = ~np.all(np.isnan(X_train), axis=0)
    valid_idx = np.where(valid)[0]

    X_pred = np.zeros((X_test.shape[0], F), dtype=np.float32)  # default 0.0

    if valid_idx.size == 0:
        # nothing to fit; return zeros
        return X_pred

    Xtr_v = X_train[:, valid_idx]
    Xte_v = X_test[:, valid_idx]

    imputer.fit(Xtr_v)
    out_v = imputer.transform(Xte_v).astype(np.float32)

    # place back
    X_pred[:, valid_idx] = out_v
    return X_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="results/baselines.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--corruption-prob", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    client_paths = sorted(glob.glob(os.path.join(args.clients_dir, "client_*.npz")))
    if not client_paths:
        raise RuntimeError(f"No client_*.npz found in {args.clients_dir}")

    baselines = ["mean", "median", "knn", "mice"]
    rows = []

    for cid, path in enumerate(client_paths):
        X, avail = load_client_npz(path)
        N, F = X.shape

        # deterministic split per client
        rng = np.random.RandomState(args.seed + cid)
        idx = rng.permutation(N)
        split = int((1.0 - args.val_frac) * N)
        tr_idx, va_idx = idx[:split], idx[split:]

        Xtr = X[tr_idx]
        Xva = X[va_idx]

        # deterministic corruption on observed+available entries
        obs_va = ~np.isnan(Xva)
        corrupt = deterministic_mask(obs_va, avail, p=args.corruption_prob, seed=args.seed)

        # hide corrupted entries in input
        Xva_in = Xva.copy()
        Xva_in[corrupt] = np.nan

        # IMPORTANT: evaluate only on corrupted entries (all are observed by design)
        for b in baselines:
            imp = make_imputer(b)
            Xva_pred = fit_transform_safe(imp, Xtr, Xva_in)

            rmse = rmse_on_mask(Xva, Xva_pred, corrupt)

            rows.append(
                {
                    "client_id": cid,
                    "baseline": b,
                    "rmse": rmse,
                    "N_train": len(tr_idx),
                    "N_val": len(va_idx),
                    "F": F,
                    "avail_frac": float(avail.mean()),
                    "corruption_prob": args.corruption_prob,
                    "valid_train_cols": int((~np.all(np.isnan(Xtr), axis=0)).sum()),
                }
            )

    df = pd.DataFrame(rows)
    summary = df.groupby("baseline")["rmse"].mean().reset_index().rename(columns={"rmse": "rmse_mean_over_clients"})
    print(summary)

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
