from __future__ import annotations
import argparse
import csv
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import flwr as fl

from fedhf.model import FedHFImputer
from fedhf.utils import ndarrays_from_state_dict, load_state_dict_from_ndarrays



class NPZClientDataset(Dataset):
    """Loads X (with NaNs) + availability_mask from .npz; returns (x, obs, avail)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"].astype(np.float32)  # standardized; NaNs represent missing/unavailable
        self.avail = d["availability_mask"].astype(np.float32)  # (F,)
        self.F = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        obs = (~np.isnan(row)).astype(np.float32)
        x = np.nan_to_num(row, nan=0.0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(obs), torch.from_numpy(self.avail)



def make_block_corruption(
    obs: torch.Tensor,
    avail: torch.Tensor,
    block_frac: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Block mask: choose a subset of FEATURES (same for the whole batch) and mask
    observed+available entries on those features.

    obs:   (B, F) {0,1}
    avail: (F,) {0,1} or (B,F) broadcastable
    returns corrupt mask: (B, F) bool
    """
    B, F = obs.shape
    device = obs.device
    if avail.ndim == 1:
        avail_f = avail.bool()
    else:
        avail_f = (avail[0] > 0).bool()

    obs_any = (obs > 0).any(dim=0)
    candidates = (avail_f & obs_any).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return torch.zeros((B, F), device=device, dtype=torch.bool)

    m = max(1, int(math.ceil(block_frac * candidates.numel())))
    perm = candidates[torch.randperm(candidates.numel(), generator=rng, device=device)]
    chosen = perm[:m]

    corrupt = torch.zeros((B, F), device=device, dtype=torch.bool)
    corrupt[:, chosen] = True

    eligible = (obs > 0)
    corrupt = corrupt & eligible
    return corrupt


def masked_recon_loss(pred: torch.Tensor, x_true: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor:
    """MSE on corrupted positions only."""
    if corrupt.sum() == 0:
        return pred.new_tensor(0.0)
    return ((pred - x_true) ** 2)[corrupt].mean()


@torch.no_grad()
def eval_rmse_deterministic(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    corruption_prob: float,
    seed: int,
    max_batches: int = 80,
) -> float:
    """
    Deterministic evaluation: fixed RNG seed ensures same masked positions each round.
    """
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    se_sum, n_sum = 0.0, 0
    for b, (x, obs, avail) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device)
        obs = obs.to(device)
        avail = avail.to(device)

        eligible = (obs == 1) & (avail == 1)
        rand = torch.rand(obs.shape, device=device, generator=g)
        corrupt = eligible & (rand < corruption_prob)

        if corrupt.sum() == 0:
            continue

        x_in = x.clone()
        x_in[corrupt] = 0.0
        obs_in = obs.clone()
        obs_in[corrupt] = 0.0

        pred = model(x_in, obs_in, avail, edge_index=edge_index, edge_weight=edge_weight)
        err = (pred - x) ** 2

        se_sum += float(err[corrupt].sum().item())
        n_sum += int(corrupt.sum().item())

    return math.sqrt(se_sum / n_sum) if n_sum > 0 else 0.0



# Client

class FedHFClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        model: FedHFImputer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        results_csv: str,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.results_csv = results_csv
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        if not os.path.exists(results_csv):
            with open(results_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "client_id", "train_loss", "val_rmse"])

    def get_parameters(self, config):
        return ndarrays_from_state_dict(self.model)

    def fit(self, parameters, config):
        load_state_dict_from_ndarrays(self.model, parameters)
        self.model.train()

        server_round = int(config.get("server_round", 0))
        lr = float(config.get("lr", 1e-3))
        local_epochs = int(config.get("local_epochs", 1))

        block_frac = float(config.get("block_frac", 0.25))  # 25% of available features per batch

        for g in self.opt.param_groups:
            g["lr"] = lr

        gmask = torch.Generator(device=self.device)
        gmask.manual_seed(10_000 + self.client_id * 1000 + server_round)

        last_loss = 0.0
        for _ in range(local_epochs):
            for x, obs, avail in self.train_loader:
                x = x.to(self.device)
                obs = obs.to(self.device)
                avail = avail.to(self.device)
                corrupt = make_block_corruption(obs, avail[0] if avail.ndim == 2 else avail, block_frac, gmask)

                x_in = x.clone()
                x_in[corrupt] = 0.0
                obs_in = obs.clone()
                obs_in[corrupt] = 0.0

                pred = self.model(x_in, obs_in, avail, edge_index=self.edge_index, edge_weight=self.edge_weight)
                loss = masked_recon_loss(pred, x, corrupt)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                last_loss = float(loss.detach().cpu().item())


        val_rmse = eval_rmse_deterministic(
            self.model,
            self.val_loader,
            self.device,
            self.edge_index,
            self.edge_weight,
            corruption_prob=float(config.get("eval_corruption_prob", 0.6)),
            seed=12345,
            max_batches=30,
        )

        with open(self.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([server_round, self.client_id, last_loss, val_rmse])

        return ndarrays_from_state_dict(self.model), len(self.train_loader.dataset), {"train_loss": last_loss}

    def evaluate(self, parameters, config):
        load_state_dict_from_ndarrays(self.model, parameters)
        server_round = int(config.get("server_round", 0))

        rmse = eval_rmse_deterministic(
            self.model,
            self.val_loader,
            self.device,
            self.edge_index,
            self.edge_weight,
            corruption_prob=float(config.get("corruption_prob", 0.6)),
            seed=12345,
            max_batches=80,
        )

        return float(rmse), len(self.val_loader.dataset), {"rmse": float(rmse)}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing client_*.npz")
    parser.add_argument("--graph-path", type=str, default="physionet_fedhf/global_graph.npz")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    args = parser.parse_args()

    client_path = os.path.join(args.data_dir, f"client_{args.client_id}.npz")
    ds = NPZClientDataset(client_path)

    # Train/val split
    n = len(ds)
    rng = np.random.RandomState(100 + args.client_id)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_ds = Subset(ds, idx[:split])
    val_ds = Subset(ds, idx[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load global graph (edge_index + edge_weight)
    g = np.load(args.graph_path)
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long, device=device)
    edge_weight = torch.tensor(g["edge_weight"], dtype=torch.float32, device=device)

    model = FedHFImputer(n_features=ds.F, d_model=args.d_model, n_layers=args.n_layers).to(device)

    results_csv = os.path.join("results", f"client_{args.client_id}.csv")

    client = FedHFClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        edge_index=edge_index,
        edge_weight=edge_weight,
        results_csv=results_csv,
    )

    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
