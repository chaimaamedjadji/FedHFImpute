import os
import numpy as np

def make_client(
    out_dir: str,
    client_id: int,
    n_clients: int,
    n_samples: int = 6000,
    n_features_global: int = 60,
    latent_dim: int = 6,
    feature_keep_ratio: float = 0.6,
    missing_prob: float = 0.15,
    seed: int = 123,
):
    rng = np.random.default_rng(seed + client_id)
    avail = np.zeros(n_features_global, dtype=np.float32)
    keep = rng.choice(n_features_global, size=int(n_features_global * feature_keep_ratio), replace=False)
    avail[keep] = 1.0
    Z = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)
    W = (rng.normal(size=(latent_dim, n_features_global)).astype(np.float32) * 0.8)
    X_full = (Z @ W + rng.normal(size=(n_samples, n_features_global)).astype(np.float32) * 0.25)
    X = X_full.copy()
    X[:, avail == 0] = np.nan
    miss = rng.uniform(size=X.shape) < missing_prob
    miss[:, avail == 0] = True
    X[miss] = np.nan

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"client_{client_id}.npz"),
        X=X.astype(np.float32),
        availability_mask=avail.astype(np.float32),
    )
    print(f"[OK] Saved client_{client_id}.npz  avail={int(avail.sum())}/{n_features_global}")

if __name__ == "__main__":
    out_dir = "data"
    n_clients = 5
    for cid in range(n_clients):
        make_client(out_dir, cid, n_clients)
