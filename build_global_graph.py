import argparse
import glob
import os
import numpy as np


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def nan_to_num(X, v=0.0):
    Y = X.copy()
    Y[np.isnan(Y)] = v
    return Y


def compute_corr_matrix_pooled(client_paths, F):
    """
    Compute pooled correlation using pairwise complete observations:
      corr(i,j) = cov(i,j)/(std_i std_j)
    using sums over observed pairs across all clients.
    """
    sum_x = np.zeros(F, dtype=np.float64)
    sum_x2 = np.zeros(F, dtype=np.float64)
    cnt_x = np.zeros(F, dtype=np.float64)

    sum_xy = np.zeros((F, F), dtype=np.float64)
    cnt_xy = np.zeros((F, F), dtype=np.float64)

    for p in client_paths:
        d = load_npz(p)
        X = d["X"].astype(np.float64)  # (N, F)
        m = ~np.isnan(X)
        X0 = nan_to_num(X, 0.0)
        sum_x += (X0 * m).sum(axis=0)
        sum_x2 += ((X0 ** 2) * m).sum(axis=0)
        cnt_x += m.sum(axis=0)
        cnt_xy += m.T @ m
        sum_xy += X0.T @ X0

    mean = sum_x / np.maximum(cnt_x, 1.0)
    var = sum_x2 / np.maximum(cnt_x, 1.0) - mean ** 2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    exy = sum_xy / np.maximum(cnt_xy, 1.0)
    cov = exy - mean[:, None] * mean[None, :]

    corr = cov / (std[:, None] * std[None, :])
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)

    return corr.astype(np.float32)


def build_topk_graph(corr, topk=12, min_abs_corr=0.05):
    F = corr.shape[0]
    edges_src = []
    edges_dst = []
    weights = []

    abs_corr = np.abs(corr)

    for i in range(F):
        idx = np.argsort(-abs_corr[i])
        chosen = []
        for j in idx:
            if j == i:
                continue
            if abs_corr[i, j] < min_abs_corr:
                break
            chosen.append(j)
            if len(chosen) >= topk:
                break

        for j in chosen:
            edges_src.append(i)
            edges_dst.append(j)
            weights.append(float(abs_corr[i, j]))

    edge_index = np.vstack([edges_src, edges_dst]).astype(np.int64)
    edge_weight = np.array(weights, dtype=np.float32)

    return edge_index, edge_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True, help="Folder containing client_*.npz")
    ap.add_argument("--out", required=True, help="Output path global_graph.npz")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--min-abs-corr", type=float, default=0.05)
    args = ap.parse_args()

    client_paths = sorted(glob.glob(os.path.join(args.clients_dir, "client_*.npz")))
    if not client_paths:
        raise RuntimeError(f"No client_*.npz found in {args.clients_dir}")

    first = load_npz(client_paths[0])
    F = first["X"].shape[1]
    print(f"[INFO] Found {len(client_paths)} clients, F={F}")

    print("[INFO] Computing pooled correlation matrix (this may take a bit)...")
    corr = compute_corr_matrix_pooled(client_paths, F)

    print("[INFO] Building top-k graph...")
    edge_index, edge_weight = build_topk_graph(
        corr, topk=args.topk, min_abs_corr=args.min_abs_corr
    )

    if edge_index.size > 0:
        mx = int(edge_index.max())
        mn = int(edge_index.min())
        assert 0 <= mn and mx < F, f"Invalid edge index range [{mn}, {mx}] for F={F}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, edge_index=edge_index, edge_weight=edge_weight, num_nodes=F)

    print(f"[OK] Saved graph to {args.out}")
    print(f"     Edges: {edge_index.shape[1]}  topk={args.topk}  min_abs_corr={args.min_abs_corr}")


if __name__ == "__main__":
    main()
