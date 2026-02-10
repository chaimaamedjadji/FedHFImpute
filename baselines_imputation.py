#!/usr/bin/env python3
"""
SOTA + Federated Imputation Baselines for FedHF-Impute datasets

LOCAL (per-client) METHODS
-------------------------
- MICE (IterativeImputer)
- Denoising Autoencoder (DAE)
- GAIN
- Transformer-based Imputer (TabTransformer-style)

FEDERATED METHODS (global model/statistics trained across clients)
-----------------------------------------------------------------
- Fed-Mean         : aggregate global feature means (raw-space), broadcast
- Fed-DAE (FedAvg) : federated denoising autoencoder via FedAvg
- Fed-Transformer  : federated transformer imputer via FedAvg

DATA FORMAT (IMPORTANT)
----------------------
Each client file: client_k.npz
  - X                  : (N_k, F) local data with NaNs
  - availability_mask  : (F,) {0,1} (features available at this client)

Global test file: test.npz
  - X                  : (N_test, F)

EVALUATION
----------
- For each client, we standardize using THAT CLIENT's train stats (mu,sigma)
- We corrupt validation only on entries that are:
    observed AND available
- RMSE computed ONLY on corrupted entries

NOTES
-----
- Fed-Mean is computed in RAW space across clients, then converted into each client's
  standardized space during evaluation via: (global_raw_mean - client_mu) / client_sigma.
- Federated deep models are trained on per-client standardized data with the same training
  corruption scheme as the local variants.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def obs_mask(X: np.ndarray):
    return (~np.isnan(X)).astype(np.float32)

def nan_to_num(X: np.ndarray, val: float = 0.0):
    Y = X.copy()
    Y[np.isnan(Y)] = val
    return Y

def standardize_fit(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)

def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu[None, :]) / sigma[None, :]

def corrupt_validation(X: np.ndarray, a: np.ndarray, p: float, seed: int):
    rng = np.random.RandomState(seed)
    m = ~np.isnan(X)
    a = a.astype(bool)

    eligible = m & a[None, :]
    mask = (rng.rand(*X.shape) < p) & eligible

    X_in = X.copy()
    X_in[mask] = np.nan
    return X_in, mask

def rmse_on_mask(x_true: np.ndarray, x_pred: np.ndarray, mask: np.ndarray):
    diff = (x_pred - x_true)[mask]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else float("nan")

def get_client_paths(clients_dir: str):
    return sorted(glob.glob(f"{clients_dir}/client_*.npz"))

def weighted_fedavg(state_dicts, weights):
    """Weighted average of PyTorch state_dicts (CPU tensors expected)."""
    out = {}
    total = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
    keys = state_dicts[0].keys()
    for k in keys:
        out[k] = sum(sd[k] * (w / total) for sd, w in zip(state_dicts, weights))
    return out


# =========================
# MICE (Local)
# =========================

def impute_mice(Xtr, Xva_in, seed):
    keep = ~np.all(np.isnan(Xtr), axis=0)
    imp = IterativeImputer(
        max_iter=10,
        random_state=seed,
        initial_strategy="mean",
        sample_posterior=False,
    )
    imp.fit(Xtr[:, keep])
    pred = np.zeros_like(Xva_in)
    pred[:, keep] = imp.transform(Xva_in[:, keep])
    return pred


# =========================
# DAE (Local + Federated)
# =========================

class DAE(nn.Module):
    def __init__(self, F, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * F, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, F),
        )

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=-1))

def train_dae_local(Xtr, a, seed, device):
    set_seed(seed)
    Xtr = Xtr.astype(np.float32)
    m = obs_mask(Xtr)
    x0 = nan_to_num(Xtr)

    F = Xtr.shape[1]
    a = a.astype(np.float32)

    model = DAE(F).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(80):
        xb = torch.from_numpy(x0).to(device)
        mb = torch.from_numpy(m).to(device)
        ab = torch.from_numpy(a).to(device)

        eligible = (mb > 0.5) & (ab[None, :] > 0.5)
        Omega = (torch.rand_like(xb) < 0.3) & eligible

        x_in = xb.clone()
        m_in = mb.clone()
        x_in[Omega] = 0.0
        m_in[Omega] = 0.0

        pred = model(x_in, m_in)
        loss = ((pred - xb) ** 2)[Omega].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model

@torch.no_grad()
def predict_dae(model, X, device):
    X = X.astype(np.float32)
    return model(
        torch.from_numpy(nan_to_num(X)).to(device),
        torch.from_numpy(obs_mask(X)).to(device)
    ).cpu().numpy()


# =========================
# GAIN (Local)
# =========================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def train_gain_local(Xtr, a, seed, device):
    set_seed(seed)
    Xtr = Xtr.astype(np.float32)
    m = obs_mask(Xtr)
    x0 = nan_to_num(Xtr)

    F = Xtr.shape[1]
    a = a.astype(np.float32)

    G = MLP(2 * F, F).to(device)
    D = MLP(2 * F, F).to(device)

    optG = optim.Adam(G.parameters(), lr=1e-3)
    optD = optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(200):
        xb = torch.from_numpy(x0).to(device)
        mb = torch.from_numpy(m).to(device)
        ab = torch.from_numpy(a).to(device)

        mb_eff = mb * (ab[None, :] > 0.5)

        # ===== Generator forward =====
        z = torch.rand_like(xb)
        x_tilde = xb * mb_eff + z * (1 - mb_eff)

        G_out = G(torch.cat([x_tilde, mb_eff], dim=-1))
        x_hat = xb * mb_eff + G_out * (1 - mb_eff)

        # ===== Discriminator step =====
        hint = (torch.rand_like(mb_eff) < 0.9).float() * mb_eff
        D_in = torch.cat([x_hat.detach(), hint], dim=-1)
        D_out = D(D_in)

        d_loss = (bce(D_out, mb_eff) * (ab[None, :] > 0.5)).mean()

        optD.zero_grad()
        d_loss.backward()
        optD.step()

        # ===== Generator step (recompute D output!) =====
        D_in_G = torch.cat([x_hat, hint], dim=-1)
        D_out_G = D(D_in_G)

        g_adv = (bce(D_out_G, torch.ones_like(mb_eff)) * (1 - mb_eff)).mean()
        rec = ((G_out - xb) ** 2 * mb_eff).mean()
        g_loss = g_adv + 10 * rec

        optG.zero_grad()
        g_loss.backward()
        optG.step()

    return G

@torch.no_grad()
def predict_gain(G, X, a, device):
    X = X.astype(np.float32)
    mb = obs_mask(X)
    xb = nan_to_num(X)
    ab = a.astype(np.float32)

    mb_eff = mb * (ab[None, :] > 0.5)
    z = np.random.rand(*xb.shape).astype(np.float32)
    x_tilde = xb * mb_eff + z * (1 - mb_eff)

    inp = np.concatenate([x_tilde, mb_eff], axis=-1)  # (N, 2F)
    return G(torch.from_numpy(inp).to(device)).cpu().numpy()


# =========================
# Transformer Imputer (Local + Federated)
# =========================

class TransformerImputer(nn.Module):
    def __init__(self, F, d=128):
        super().__init__()
        self.emb = nn.Embedding(F, d)
        self.val = nn.Linear(1, d)
        self.flag = nn.Linear(2, d)
        enc = nn.TransformerEncoderLayer(d, 4, 256, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 3)
        self.out = nn.Linear(d, 1)

    def forward(self, x, m, a):
        B, F = x.shape
        e = self.emb(torch.arange(F, device=x.device)).unsqueeze(0).expand(B, F, -1)
        v = self.val(x.unsqueeze(-1))
        f = self.flag(torch.stack([m, a.unsqueeze(0).expand_as(m)], -1))
        h = self.tr(e + v + f)
        return self.out(h).squeeze(-1)

def train_transformer_local(Xtr, a, seed, device):
    set_seed(seed)
    Xtr = Xtr.astype(np.float32)
    m = obs_mask(Xtr)
    x0 = nan_to_num(Xtr)

    F = Xtr.shape[1]
    a = a.astype(np.float32)

    model = TransformerImputer(F).to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for _ in range(120):
        xb = torch.from_numpy(x0).to(device)
        mb = torch.from_numpy(m).to(device)
        ab = torch.from_numpy(a).to(device)

        eligible = (mb > 0.5) & (ab[None, :] > 0.5)
        Omega = (torch.rand_like(xb) < 0.3) & eligible

        x_in = xb.clone()
        m_in = mb.clone()
        x_in[Omega] = 0.0
        m_in[Omega] = 0.0

        pred = model(x_in, m_in, ab)
        loss = ((pred - xb) ** 2)[Omega].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model

@torch.no_grad()
def predict_transformer(model, X, a, device):
    return model(
        torch.from_numpy(nan_to_num(X)).to(device),
        torch.from_numpy(obs_mask(X)).to(device),
        torch.from_numpy(a.astype(np.float32)).to(device)
    ).cpu().numpy()


# =========================
# Fed-Mean (Global stats)
# =========================

def fit_fed_mean_raw(clients_paths):
    """Compute global feature mean in RAW space over (observed & available) entries."""
    sumv = None
    cntv = None

    for p in clients_paths:
        d = load_npz(p)
        X = d["X"].astype(np.float32)
        a = d["availability_mask"].astype(bool)

        obs = ~np.isnan(X)
        elig = obs & a[None, :]

        X0 = nan_to_num(X, 0.0)
        s = (X0 * elig).sum(axis=0).astype(np.float64)
        c = elig.sum(axis=0).astype(np.float64)

        if sumv is None:
            sumv = s
            cntv = c
        else:
            sumv += s
            cntv += c

    mu_raw = (sumv / np.maximum(cntv, 1.0)).astype(np.float32)
    return mu_raw

def predict_fed_mean_in_client_zspace(mu_raw_global, client_mu_raw, client_sigma_raw, Xt_in_z):
    mu_z = (mu_raw_global - client_mu_raw) / client_sigma_raw  # (F,)
    pred = nan_to_num(Xt_in_z).copy()
    miss = np.isnan(Xt_in_z)
    pred = np.where(miss, mu_z[None, :], pred)
    return pred

# =========================
# Federated training loops (FedAvg)
# =========================

def fed_train_dae(clients_paths, seed, device, rounds=20, local_epochs=1, lr=1e-3):
    set_seed(seed)

    d0 = load_npz(clients_paths[0])
    F = d0["X"].shape[1]
    global_model = DAE(F).to(device)

    for _ in range(rounds):
        local_states = []
        weights = []

        for cid, p in enumerate(clients_paths):
            d = load_npz(p)
            Xtr_raw = d["X"].astype(np.float32)
            a = d["availability_mask"].astype(np.float32)

            # local z-space (per client)
            mu, sigma = standardize_fit(Xtr_raw)
            Xtr = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)

            m = obs_mask(Xtr)
            x0 = nan_to_num(Xtr)

            local_model = DAE(F).to(device)
            local_model.load_state_dict(global_model.state_dict())
            opt = optim.Adam(local_model.parameters(), lr=lr)

            for _e in range(local_epochs):
                xb = torch.from_numpy(x0).to(device)
                mb = torch.from_numpy(m).to(device)
                ab = torch.from_numpy(a).to(device)

                eligible = (mb > 0.5) & (ab[None, :] > 0.5)
                Omega = (torch.rand_like(xb) < 0.3) & eligible

                x_in = xb.clone()
                m_in = mb.clone()
                x_in[Omega] = 0.0
                m_in[Omega] = 0.0

                pred = local_model(x_in, m_in)
                loss = ((pred - xb) ** 2)[Omega].mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

            local_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(Xtr_raw.shape[0])

        new_state = weighted_fedavg(local_states, weights)
        global_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})

    return global_model

def fed_train_transformer(clients_paths, seed, device, rounds=20, local_epochs=1, lr=2e-4):
    set_seed(seed)

    d0 = load_npz(clients_paths[0])
    F = d0["X"].shape[1]
    global_model = TransformerImputer(F).to(device)

    for _ in range(rounds):
        local_states = []
        weights = []

        for cid, p in enumerate(clients_paths):
            d = load_npz(p)
            Xtr_raw = d["X"].astype(np.float32)
            a = d["availability_mask"].astype(np.float32)

            mu, sigma = standardize_fit(Xtr_raw)
            Xtr = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)

            m = obs_mask(Xtr)
            x0 = nan_to_num(Xtr)

            local_model = TransformerImputer(F).to(device)
            local_model.load_state_dict(global_model.state_dict())
            opt = optim.AdamW(local_model.parameters(), lr=lr)

            for _e in range(local_epochs):
                xb = torch.from_numpy(x0).to(device)
                mb = torch.from_numpy(m).to(device)
                ab = torch.from_numpy(a).to(device)

                eligible = (mb > 0.5) & (ab[None, :] > 0.5)
                Omega = (torch.rand_like(xb) < 0.3) & eligible

                x_in = xb.clone()
                m_in = mb.clone()
                x_in[Omega] = 0.0
                m_in[Omega] = 0.0

                pred = local_model(x_in, m_in, ab)
                loss = ((pred - xb) ** 2)[Omega].mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

            local_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(Xtr_raw.shape[0])

        new_state = weighted_fedavg(local_states, weights)
        global_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})

    return global_model


# =========================
# Runners
# =========================

def run_one_client_local(client_npz, test_npz, methods, p, seed, device):
    d = load_npz(client_npz)
    Xtr_raw = d["X"]
    a = d["availability_mask"]

    Xt_raw = load_npz(test_npz)["X"]

    # client-specific standardization
    mu, sigma = standardize_fit(Xtr_raw)
    Xtr = standardize_apply(Xtr_raw, mu, sigma)
    Xt = standardize_apply(Xt_raw, mu, sigma)

    Xt_in, mask = corrupt_validation(Xt, a, p, seed)

    out = []
    for m in methods:
        if m == "mice":
            pred = impute_mice(Xtr, Xt_in, seed)
        elif m == "dae":
            pred = predict_dae(train_dae_local(Xtr, a, seed, device), Xt_in, device)
        elif m == "gain":
            pred = predict_gain(train_gain_local(Xtr, a, seed, device), Xt_in, a, device)
        elif m == "transformer":
            pred = predict_transformer(train_transformer_local(Xtr, a, seed, device), Xt_in, a, device)
        else:
            raise ValueError(f"Unknown local method: {m}")
        out.append((m, rmse_on_mask(Xt, pred, mask)))
    return out

def run_federated(clients_dir, test_npz, fed_methods, p, seed, device, rounds=20, local_epochs=1):
    clients_paths = get_client_paths(clients_dir)
    Xt_raw = load_npz(test_npz)["X"]

    # Train global models/stats once
    fed_models = {}
    mu_raw_global = None

    if "fed_mean" in fed_methods:
        mu_raw_global = fit_fed_mean_raw(clients_paths)

    if "fed_dae" in fed_methods:
        fed_models["fed_dae"] = fed_train_dae(
            clients_paths, seed=seed, device=device, rounds=rounds, local_epochs=local_epochs
        )

    if "fed_transformer" in fed_methods:
        fed_models["fed_transformer"] = fed_train_transformer(
            clients_paths, seed=seed, device=device, rounds=rounds, local_epochs=local_epochs
        )

    # Evaluate per client (using that client's mu/sigma)
    rows = []
    for cid, cp in enumerate(clients_paths):
        d = load_npz(cp)
        Xtr_raw = d["X"]
        a = d["availability_mask"]

        client_mu, client_sigma = standardize_fit(Xtr_raw)
        Xt = standardize_apply(Xt_raw, client_mu, client_sigma)

        Xt_in, mask = corrupt_validation(Xt, a, p, seed + cid)

        for m in fed_methods:
            if m == "fed_mean":
                pred = predict_fed_mean_in_client_zspace(
                    mu_raw_global, client_mu, client_sigma, Xt_in
                )
            elif m == "fed_dae":
                pred = predict_dae(fed_models["fed_dae"], Xt_in, device)
            elif m == "fed_transformer":
                pred = predict_transformer(fed_models["fed_transformer"], Xt_in, a, device)
            else:
                raise ValueError(f"Unknown federated method: {m}")

            rows.append({
                "client_id": cid,
                "method": m,
                "rmse": rmse_on_mask(Xt, pred, mask),
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--out-csv", required=True)

    ap.add_argument(
        "--methods",
        default="mice,dae,gain,transformer,fed_mean,fed_dae,fed_transformer",
        help="Comma-separated. Local: mice,dae,gain,transformer. Federated: fed_mean,fed_dae,fed_transformer"
    )

    ap.add_argument("--corruption-prob", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--device", default="cpu")

    # Federated training knobs
    ap.add_argument("--fed-rounds", type=int, default=20)
    ap.add_argument("--fed-local-epochs", type=int, default=1)

    args = ap.parse_args()
    device = torch.device(args.device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    local_methods = [m for m in methods if not m.startswith("fed_")]
    fed_methods = [m for m in methods if m.startswith("fed_")]

    rows = []

    # ----- Local baselines -----
    client_paths = get_client_paths(args.clients_dir)
    for cid, pth in enumerate(client_paths):
        if len(local_methods) == 0:
            break
        for name, rmse in run_one_client_local(
            pth, args.test_npz, local_methods, args.corruption_prob, args.seed + cid, device
        ):
            rows.append({
                "client_id": cid,
                "method": name,
                "rmse": rmse,
            })

    # ----- Federated baselines -----
    if len(fed_methods) > 0:
        fed_rows = run_federated(
            clients_dir=args.clients_dir,
            test_npz=args.test_npz,
            fed_methods=fed_methods,
            p=args.corruption_prob,
            seed=args.seed,
            device=device,
            rounds=args.fed_rounds,
            local_epochs=args.fed_local_epochs,
        )
        rows.extend(fed_rows)

    df = pd.DataFrame(rows)
    ensure_dir(args.out_csv)
    df.to_csv(args.out_csv, index=False)

    print(df.groupby("method")["rmse"].agg(["mean", "std", "count"]).sort_values("mean"))

if __name__ == "__main__":
    main()
