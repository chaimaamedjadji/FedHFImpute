from __future__ import annotations
import argparse
import math
import numpy as np
import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

from fedhf.dataset import HeteroNPZDataset
from fedhf.model import FedHFImputer
from fedhf.utils import ndarrays_from_state_dict, load_state_dict_from_ndarrays, build_structured_graph


def masked_recon_loss(pred, x_true, obs_mask, avail_mask, corruption_prob: float):
    eligible = (obs_mask == 1) & (avail_mask == 1)
    rand = torch.rand_like(obs_mask)
    corrupt = eligible & (rand < corruption_prob)

    if corrupt.sum() == 0:
        return pred.new_tensor(0.0)

    mse = (pred - x_true) ** 2
    return mse[corrupt].mean()


@torch.no_grad()
def eval_rmse(model, loader, device, edge_index, corruption_prob=0.5, max_batches=50):
    model.eval()
    se_sum, n_sum = 0.0, 0

    for b, (x, obs, avail) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device)
        obs = obs.to(device)
        avail = avail.to(device)

        eligible = (obs == 1) & (avail == 1)
        rand = torch.rand_like(obs)
        corrupt = eligible & (rand < corruption_prob)

        if corrupt.sum() == 0:
            continue

        x_in = x.clone()
        x_in[corrupt] = 0.0
        obs_in = obs.clone()
        obs_in[corrupt] = 0.0

        pred = model(x_in, obs_in, avail, edge_index=edge_index)
        err = (pred - x) ** 2

        se_sum += float(err[corrupt].sum().item())
        n_sum += int(corrupt.sum().item())

    if n_sum == 0:
        return 0.0
    return math.sqrt(se_sum / n_sum)


class FedHFClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, edge_index):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.edge_index = edge_index
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_parameters(self, config):
        return ndarrays_from_state_dict(self.model)

    def fit(self, parameters, config):
        load_state_dict_from_ndarrays(self.model, parameters)
        self.model.train()

        lr = float(config.get("lr", 1e-3))
        local_epochs = int(config.get("local_epochs", 1))
        corruption_prob = float(config.get("corruption_prob", 0.5))

        for g in self.opt.param_groups:
            g["lr"] = lr

        last_loss = 0.0
        for _ in range(local_epochs):
            for x, obs, avail in self.train_loader:
                x = x.to(self.device)
                obs = obs.to(self.device)
                avail = avail.to(self.device)

                pred = self.model(x, obs, avail, edge_index=self.edge_index)
                loss = masked_recon_loss(pred, x, obs, avail, corruption_prob)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                last_loss = float(loss.detach().cpu().item())

        return ndarrays_from_state_dict(self.model), len(self.train_loader.dataset), {"train_loss": last_loss}

    def evaluate(self, parameters, config):
        load_state_dict_from_ndarrays(self.model, parameters)
        corruption_prob = float(config.get("corruption_prob", 0.5))

        rmse = eval_rmse(
            self.model, self.val_loader, self.device,
            edge_index=self.edge_index,
            corruption_prob=corruption_prob,
            max_batches=80,
        )
        return float(rmse), len(self.val_loader.dataset), {"rmse": float(rmse)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    args = parser.parse_args()

    path = f"{args.data_dir}/client_{args.client_id}.npz"
    ds = HeteroNPZDataset(path)

    # ---- train/val split (fixed seed for reproducibility) ----
    n = len(ds)
    rng = np.random.RandomState(0 + args.client_id)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FedHFImputer(n_features=ds.F, d_model=args.d_model, n_layers=args.n_layers).to(device)

    # ---- deterministic structured graph built locally (Flower-safe) ----
    edge_index_np = build_structured_graph(n_nodes=ds.F, k=8)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    client = FedHFClient(model, train_dl, val_dl, device, edge_index=edge_index)

    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
