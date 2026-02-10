import numpy as np
import torch
from torch.utils.data import Dataset

class HeteroNPZDataset(Dataset):
    """
    Returns standardized x (NaNs -> 0 after standardization),
    plus obs_mask and avail_mask.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)              # (N, F) with NaNs
        self.avail = data["availability_mask"].astype(np.float32)  # (F,)
        self.F = X.shape[1]

        # Compute mean/std per feature on observed entries only AND available features
        mu = np.zeros(self.F, dtype=np.float32)
        sd = np.ones(self.F, dtype=np.float32)

        for f in range(self.F):
            if self.avail[f] == 0:
                continue
            col = X[:, f]
            obs = ~np.isnan(col)
            if obs.sum() < 5:
                continue
            mu[f] = col[obs].mean()
            sd[f] = col[obs].std() + 1e-6

        self.mu = mu
        self.sd = sd

        # Store standardized X (keep NaNs for masks, but normalized values in observed positions)
        Xn = (X - mu[None, :]) / sd[None, :]
        self.X = Xn.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        obs = (~np.isnan(row)).astype(np.float32)
        x = np.nan_to_num(row, nan=0.0).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(obs),
            torch.from_numpy(self.avail),
        )
