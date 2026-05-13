"""
Baselines for imputation under heterogeneous feature availability:
- Local: mice, vfl_knn, dae, gain, transformer
- Federated: fed_mean, fed_dae, fed_transformer, fed_featgen

Evaluation consistency:
- RMSE is computed in RAW space on the corrupted mask only.

python baselines_imputation.py \
  --clients-dir path/to/clients \
  --test-npz path/to/test.npz \
  --out-csv results/out.csv \
  --methods mice,vfl_knn,dae,gain,transformer,fed_mean,fed_dae,fed_transformer,fed_featgen \
  --corruption-prob 0.6 \
  --seed 123 \
  --device cpu \
  --fed-rounds 20 \
  --fed-local-epochs 1
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
import torch
import torch.nn as nn
import torch.optim as optim

# Utilities functions
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

def standardize_fit(X_raw: np.ndarray):
    mu = np.nanmean(X_raw, axis=0)
    sigma = np.nanstd(X_raw, axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)

def standardize_apply(X_raw: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X_raw - mu[None, :]) / sigma[None, :]

def standardize_invert(Z: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return Z * sigma[None, :] + mu[None, :]

def finalize_imputation(X_in_raw: np.ndarray, X_pred_raw: np.ndarray):
    """
    It keeps the observed entries from X_in_raw and fill NaNs with X_pred_raw.
    """
    out = X_in_raw.copy()
    miss = np.isnan(out)
    out[miss] = X_pred_raw[miss]
    return out

def corrupt_validation(X_raw: np.ndarray, a: np.ndarray, p: float, seed: int, train_obs_cols: np.ndarray = None):
    rng = np.random.RandomState(seed)
    m_test = ~np.isnan(X_raw)
    a = a.astype(bool)
    eligible = m_test & a[None, :]
    # only corrupt columns that have at least one observed value in this client's TRAIN split
    if train_obs_cols is not None:
        eligible = eligible & train_obs_cols[None, :]

    mask = (rng.rand(*X_raw.shape) < p) & eligible
    X_in = X_raw.copy()
    X_in[mask] = np.nan
    return X_in, mask

def rmse_on_mask(x_true: np.ndarray, x_pred: np.ndarray, mask: np.ndarray):
    diff = (x_pred - x_true)[mask]
    diff = diff[~np.isnan(diff)]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else float("nan")

def get_client_paths(clients_dir: str):
    return sorted(glob.glob(f"{clients_dir}/client_*.npz"))

def weighted_fedavg(state_dicts, weights):
    out = {}
    total = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
    for k in state_dicts[0].keys():
        out[k] = sum(sd[k] * (w / total) for sd, w in zip(state_dicts, weights))
    return out



# Local baselines (MICE, VFL-KNN)

def impute_mice(Xtr_raw, Xt_in_raw, a, seed):
    Xtr_raw = Xtr_raw.astype(np.float32)
    Xt_in_raw = Xt_in_raw.astype(np.float32)
    a = a.astype(bool)
    pred = Xt_in_raw.copy()
    F = pred.shape[1]
    fb = np.nanmean(Xtr_raw, axis=0).astype(np.float32)  # (F,)
    fb[np.isnan(fb)] = 0.0
    cols = np.where(a)[0]
    if cols.size == 0:
        # fill any missing values
        miss = np.isnan(pred)
        pred[miss] = np.broadcast_to(fb, pred.shape)[miss]
        return pred
    # fit MICE only on columns that have some training signal
    keep = ~np.all(np.isnan(Xtr_raw[:, cols]), axis=0)
    cols_keep = cols[keep]
    if cols_keep.size > 0:
        imp = IterativeImputer(
            max_iter=10,
            random_state=seed,
            initial_strategy="mean",
            sample_posterior=False,
        )
        imp.fit(Xtr_raw[:, cols_keep])
        Xt_imp = imp.transform(Xt_in_raw[:, cols_keep])
        miss = np.isnan(pred[:, cols_keep])
        pred[:, cols_keep][miss] = Xt_imp[miss]

    # final hard fill
    miss = np.isnan(pred)
    miss &= a[None, :]
    if miss.any():
        fb_mat = np.broadcast_to(fb, pred.shape)
        pred[miss] = fb_mat[miss]
    return pred

def impute_vfl_knn(Xtr_raw, Xt_in_raw, a, n_neighbors=5, weights="distance"):
    Xtr_raw = Xtr_raw.astype(np.float32)
    Xt_in_raw = Xt_in_raw.astype(np.float32)
    a = a.astype(bool)
    pred = Xt_in_raw.copy()
    F = pred.shape[1]
    fb = np.nanmean(Xtr_raw, axis=0).astype(np.float32)
    fb[np.isnan(fb)] = 0.0
    cols = np.where(a)[0]
    if cols.size > 0:
        keep = ~np.all(np.isnan(Xtr_raw[:, cols]), axis=0)
        cols_keep = cols[keep]
        if cols_keep.size > 0:
            imp = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imp.fit(Xtr_raw[:, cols_keep])
            Xt_imp = imp.transform(Xt_in_raw[:, cols_keep])
            miss = np.isnan(pred[:, cols_keep])
            pred[:, cols_keep][miss] = Xt_imp[miss]
    # final hard fill
    miss = np.isnan(pred)
    miss &= a[None, :]
    if miss.any():
        fb_mat = np.broadcast_to(fb, pred.shape)
        pred[miss] = fb_mat[miss]
    return pred



# DAE (Local & Federated)

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

def train_dae_local(Xtr_z, a, seed, device):
    set_seed(seed)
    Xtr_z = Xtr_z.astype(np.float32)
    m = obs_mask(Xtr_z)
    x0 = nan_to_num(Xtr_z)
    F = Xtr_z.shape[1]
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
def predict_dae(model, X_z, device):
    X_z = X_z.astype(np.float32)
    return model(
        torch.from_numpy(nan_to_num(X_z)).to(device),
        torch.from_numpy(obs_mask(X_z)).to(device)
    ).cpu().numpy()



# GAIN (Local)

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

def train_gain_local(Xtr_z, a, seed, device):
    set_seed(seed)
    Xtr_z = Xtr_z.astype(np.float32)
    m = obs_mask(Xtr_z)
    x0 = nan_to_num(Xtr_z)

    F = Xtr_z.shape[1]
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

        z = torch.rand_like(xb)
        x_tilde = xb * mb_eff + z * (1 - mb_eff)

        G_out = G(torch.cat([x_tilde, mb_eff], dim=-1))
        x_hat = xb * mb_eff + G_out * (1 - mb_eff)

        hint = (torch.rand_like(mb_eff) < 0.9).float() * mb_eff
        D_in = torch.cat([x_hat.detach(), hint], dim=-1)
        D_out = D(D_in)

        d_loss = (bce(D_out, mb_eff) * (ab[None, :] > 0.5)).mean()

        optD.zero_grad()
        d_loss.backward()
        optD.step()

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
def predict_gain(G, X_z, a, device):
    """
    Returns x_hat (imputed matrix) in z-space.
    """
    X_z = X_z.astype(np.float32)
    mb = obs_mask(X_z)
    xb = nan_to_num(X_z)
    ab = a.astype(np.float32)

    mb_eff = mb * (ab[None, :] > 0.5)
    z = np.random.rand(*xb.shape).astype(np.float32)
    x_tilde = xb * mb_eff + z * (1 - mb_eff)

    inp = np.concatenate([x_tilde, mb_eff], axis=-1)
    G_out = G(torch.from_numpy(inp).to(device)).cpu().numpy()

    x_hat = xb * mb_eff + G_out * (1 - mb_eff)
    return x_hat


# -------------------------
# Transformer Imputer (Local + Federated)
# -------------------------

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

def train_transformer_local(Xtr_z, a, seed, device):
    set_seed(seed)
    Xtr_z = Xtr_z.astype(np.float32)
    m = obs_mask(Xtr_z)
    x0 = nan_to_num(Xtr_z)

    F = Xtr_z.shape[1]
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
def predict_transformer(model, X_z, a, device):
    return model(
        torch.from_numpy(nan_to_num(X_z)).to(device),
        torch.from_numpy(obs_mask(X_z)).to(device),
        torch.from_numpy(a.astype(np.float32)).to(device)
    ).cpu().numpy()


# -------------------------
# Fed-Mean (Global stats in RAW space)
# -------------------------

def fit_fed_mean_raw(clients_paths):
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

def predict_fed_mean_raw(mu_raw_global, Xt_in_raw):
    """
    Correct broadcasting under boolean mask.
    """
    pred = Xt_in_raw.copy()
    miss = np.isnan(pred)  # (N, F)
    mu_mat = np.broadcast_to(mu_raw_global, pred.shape)  # (N, F)
    pred[miss] = mu_mat[miss]
    return pred


# -------------------------
# Federated training: DAE / Transformer
# -------------------------

def fed_train_dae(clients_paths, seed, device, rounds=20, local_epochs=1, lr=1e-3):
    set_seed(seed)

    d0 = load_npz(clients_paths[0])
    F = d0["X"].shape[1]
    global_model = DAE(F).to(device)

    for _ in range(rounds):
        local_states = []
        weights = []

        for p in clients_paths:
            d = load_npz(p)
            Xtr_raw = d["X"].astype(np.float32)
            a = d["availability_mask"].astype(np.float32)

            mu, sigma = standardize_fit(Xtr_raw)
            Xtr_z = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)

            m = obs_mask(Xtr_z)
            x0 = nan_to_num(Xtr_z)

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

        for p in clients_paths:
            d = load_npz(p)
            Xtr_raw = d["X"].astype(np.float32)
            a = d["availability_mask"].astype(np.float32)

            mu, sigma = standardize_fit(Xtr_raw)
            Xtr_z = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)

            m = obs_mask(Xtr_z)
            x0 = nan_to_num(Xtr_z)

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


# -------------------------
# FedFeatGen (Federated Feature Generator baseline)
# -------------------------

class FeatGenNet(nn.Module):
    """
    Feature generator / translator style baseline:
    masked reconstruction autoencoder with bottleneck.
    """
    def __init__(self, F, hidden=256, bottleneck=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(2 * F, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, F),
        )

    def forward(self, x, m):
        z = self.enc(torch.cat([x, m], dim=-1))
        return self.dec(z)

def fed_train_featgen(clients_paths, seed, device, rounds=20, local_epochs=1, lr=1e-3,
                      hidden=256, bottleneck=64):
    set_seed(seed)

    d0 = load_npz(clients_paths[0])
    F = d0["X"].shape[1]
    global_model = FeatGenNet(F, hidden=hidden, bottleneck=bottleneck).to(device)

    for _ in range(rounds):
        local_states = []
        weights = []

        for p in clients_paths:
            d = load_npz(p)
            Xtr_raw = d["X"].astype(np.float32)
            a = d["availability_mask"].astype(np.float32)

            mu, sigma = standardize_fit(Xtr_raw)
            Xtr_z = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)

            m = obs_mask(Xtr_z)
            x0 = nan_to_num(Xtr_z)

            local_model = FeatGenNet(F, hidden=hidden, bottleneck=bottleneck).to(device)
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

@torch.no_grad()
def predict_featgen(model, X_z, device):
    X_z = X_z.astype(np.float32)
    return model(
        torch.from_numpy(nan_to_num(X_z)).to(device),
        torch.from_numpy(obs_mask(X_z)).to(device)
    ).cpu().numpy()


# -------------------------
# Simulation / Evaluation (RAW RMSE)
# -------------------------

def run_one_client_local(client_npz, test_npz, methods, p, seed, device):
    d = load_npz(client_npz)
    Xtr_raw = d["X"].astype(np.float32)
    a = d["availability_mask"]

    Xt_raw = load_npz(test_npz)["X"].astype(np.float32)
    train_counts = np.sum(~np.isnan(Xtr_raw), axis=0)  # (F,)
    train_good = train_counts >= 20

    Xt_in_raw, mask = corrupt_validation(Xt_raw, a, p, seed, train_obs_cols=train_good)
    #Xt_in_raw, mask = corrupt_validation(Xt_raw, a, p, seed, train_obs_cols=train_obs_cols)
    # Corrupt RAW test consistently
    #Xt_in_raw, mask = corrupt_validation(Xt_raw, a, p, seed)

    # Client-specific standardization (for NN methods)
    mu, sigma = standardize_fit(Xtr_raw)
    Xtr_z = standardize_apply(Xtr_raw, mu, sigma).astype(np.float32)
    Xt_in_z = standardize_apply(Xt_in_raw, mu, sigma).astype(np.float32)

    out = []
    for m in methods:
        if m == "mice":
            pred_raw = impute_mice(Xtr_raw, Xt_in_raw, a, seed)

        elif m == "vfl_knn":
            pred_raw = impute_vfl_knn(Xtr_raw, Xt_in_raw, a, n_neighbors=5)

        elif m == "dae":
            model = train_dae_local(Xtr_z, a, seed, device)
            pred_z = predict_dae(model, Xt_in_z, device)
            pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z, mu, sigma))

        elif m == "gain":
            G = train_gain_local(Xtr_z, a, seed, device)
            pred_z_full = predict_gain(G, Xt_in_z, a, device)  # full imputed in z-space
            pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z_full, mu, sigma))

        elif m == "transformer":
            model = train_transformer_local(Xtr_z, a, seed, device)
            pred_z = predict_transformer(model, Xt_in_z, a, device)
            pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z, mu, sigma))

        else:
            raise ValueError(f"Unknown local method: {m}")

        out.append((m, rmse_on_mask(Xt_raw, pred_raw, mask)))

    return out

def run_federated(clients_dir, test_npz, fed_methods, p, seed, device, rounds=20, local_epochs=1):
    clients_paths = get_client_paths(clients_dir)
    Xt_raw = load_npz(test_npz)["X"].astype(np.float32)

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

    if "fed_featgen" in fed_methods:
        fed_models["fed_featgen"] = fed_train_featgen(
            clients_paths, seed=seed, device=device, rounds=rounds, local_epochs=local_epochs
        )

    rows = []
    for cid, cp in enumerate(clients_paths):
        d = load_npz(cp)
        Xtr_raw = d["X"].astype(np.float32)
        a = d["availability_mask"]
        train_counts = np.sum(~np.isnan(Xtr_raw), axis=0)  # (F,)
        train_good = train_counts >= 20

        Xt_in_raw, mask = corrupt_validation(Xt_raw, a, p, seed, train_obs_cols=train_good)
        # Corrupt RAW test consistently
        #Xt_in_raw, mask = corrupt_validation(Xt_raw, a, p, seed + cid)

        # Client-specific standardization for NN inference
        client_mu, client_sigma = standardize_fit(Xtr_raw)
        Xt_in_z = standardize_apply(Xt_in_raw, client_mu, client_sigma).astype(np.float32)

        for m in fed_methods:
            if m == "fed_mean":
                pred_raw = predict_fed_mean_raw(mu_raw_global, Xt_in_raw)

            elif m == "fed_dae":
                pred_z = predict_dae(fed_models["fed_dae"], Xt_in_z, device)
                pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z, client_mu, client_sigma))

            elif m == "fed_transformer":
                pred_z = predict_transformer(fed_models["fed_transformer"], Xt_in_z, a, device)
                pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z, client_mu, client_sigma))

            elif m == "fed_featgen":
                pred_z = predict_featgen(fed_models["fed_featgen"], Xt_in_z, device)
                pred_raw = finalize_imputation(Xt_in_raw, standardize_invert(pred_z, client_mu, client_sigma))

            else:
                raise ValueError(f"Unknown federated method: {m}")

            rows.append({
                "client_id": cid,
                "method": m,
                "rmse_raw": rmse_on_mask(Xt_raw, pred_raw, mask),
            })

    return rows


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--out-csv", required=True)

    ap.add_argument(
        "--methods",
        default="mice,vfl_knn,dae,gain,transformer,fed_mean,fed_dae,fed_transformer,fed_featgen",
        help="Comma-separated. Local: mice,vfl_knn,dae,gain,transformer. "
             "Federated: fed_mean,fed_dae,fed_transformer,fed_featgen"
    )

    ap.add_argument("--corruption-prob", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")

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

        results = run_one_client_local(
            pth, args.test_npz, local_methods, args.corruption_prob, args.seed + cid, device
        )
        for name, rmse_raw in results:
            rows.append({
                "client_id": cid,
                "method": name,
                "rmse_raw": rmse_raw,
            })

    # ----- Federated baselines -----
    if len(fed_methods) > 0:
        rows.extend(run_federated(
            clients_dir=args.clients_dir,
            test_npz=args.test_npz,
            fed_methods=fed_methods,
            p=args.corruption_prob,
            seed=args.seed,
            device=device,
            rounds=args.fed_rounds,
            local_epochs=args.fed_local_epochs,
        ))

    df = pd.DataFrame(rows)
    ensure_dir(args.out_csv)
    df.to_csv(args.out_csv, index=False)

    print(df.groupby("method")["rmse_raw"].agg(["mean", "std", "count"]).sort_values("mean"))

if __name__ == "__main__":
    main()
