#!/usr/bin/env python3
"""
SOTA Imputation Baselines for FedHF-Impute datasets

SUPPORTED METHODS
-----------------
- MICE (IterativeImputer)
- Denoising Autoencoder (DAE)
- GAIN
- Transformer-based Imputer (TabTransformer-style)

DATA FORMAT (IMPORTANT)
----------------------
Each client file: client_k.npz
  - X                  : (N_k, F) local data with NaNs
  - availability_mask  : (F,) {0,1}

Global test file: test.npz
  - X                  : (N_test, F)

EVALUATION
----------
- Validation corruption only on entries that are:
    observed AND available
- RMSE computed ONLY on corrupted entries
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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def obs_mask(X):
    return (~np.isnan(X)).astype(np.float32)

def nan_to_num(X, val=0.0):
    Y = X.copy()
    Y[np.isnan(Y)] = val
    return Y

def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu[None, :]) / sigma[None, :]

def corrupt_validation(X, a, p, seed):
    rng = np.random.RandomState(seed)
    m = ~np.isnan(X)
    a = a.astype(bool)

    eligible = m & a[None, :]
    mask = (rng.rand(*X.shape) < p) & eligible

    X_in = X.copy()
    X_in[mask] = np.nan
    return X_in, mask

def rmse_on_mask(x_true, x_pred, mask):
    diff = (x_pred - x_true)[mask]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else float("nan")


# =========================
# MICE
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
# DAE
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

def train_dae(Xtr, a, seed, device):
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
# GAIN
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

def train_gain(Xtr, a, seed, device):
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

    return G(
        torch.from_numpy(np.concatenate([x_tilde, mb_eff], axis=1)).to(device)
    ).cpu().numpy()


# =========================
# Transformer Imputer
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

def train_transformer(Xtr, a, seed, device):
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
        torch.from_numpy(a).to(device)
    ).cpu().numpy()


# =========================
# Runner
# =========================

def run_one_client(client_npz, test_npz, methods, p, seed, device):
    d = load_npz(client_npz)
    Xtr = d["X"]
    a = d["availability_mask"]

    Xt = load_npz(test_npz)["X"]

    mu, sigma = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, sigma)
    Xt = standardize_apply(Xt, mu, sigma)

    Xt_in, mask = corrupt_validation(Xt, a, p, seed)

    out = []
    for m in methods:
        if m == "mice":
            pred = impute_mice(Xtr, Xt_in, seed)
        elif m == "dae":
            pred = predict_dae(train_dae(Xtr, a, seed, device), Xt_in, device)
        elif m == "gain":
            pred = predict_gain(train_gain(Xtr, a, seed, device), Xt_in, a, device)
        elif m == "transformer":
            pred = predict_transformer(train_transformer(Xtr, a, seed, device), Xt_in, a, device)
        else:
            raise ValueError(m)
        out.append((m, rmse_on_mask(Xt, pred, mask)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--methods", default="mice,dae,gain,transformer")
    ap.add_argument("--corruption-prob", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    methods = args.methods.split(",")
    rows = []

    for cid, p in enumerate(sorted(glob.glob(f"{args.clients_dir}/client_*.npz"))):
        for name, rmse in run_one_client(
            p, args.test_npz, methods, args.corruption_prob, args.seed + cid, args.device
        ):
            rows.append({
                "client_id": cid,
                "method": name,
                "rmse": rmse,
            })

    df = pd.DataFrame(rows)
    ensure_dir(args.out_csv)
    df.to_csv(args.out_csv, index=False)

    print(df.groupby("method")["rmse"].agg(["mean", "std"]))

if __name__ == "__main__":
    main()
