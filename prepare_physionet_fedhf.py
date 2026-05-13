import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import urllib.request
import zipfile

PHYSIONET_URL = "https://physionet.org/files/challenge-2012/1.0.0/set-a.zip"


def download_dataset(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "set-a.zip"

    if not zip_path.exists():
        print("Downloading PhysioNet dataset...")
        urllib.request.urlretrieve(PHYSIONET_URL, zip_path)

    extract_dir = raw_dir / "set-a"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)

    return extract_dir



def load_physionet_data(folder):
    files = list(Path(folder).glob("*.txt"))

    all_samples = []

    for f in files:
        df = pd.read_csv(f)

        df = df.sort_values("Time")
        df = df.drop_duplicates("Parameter", keep="last")

        pivot = df.pivot_table(index=None, columns="Parameter", values="Value")

        all_samples.append(pivot)

    data = pd.concat(all_samples, ignore_index=True)

    return data



def federated_split(X, n_clients=5, keep_ratio=0.6, seed=42):
    np.random.seed(seed)
    N, F = X.shape

    client_data = []

    for cid in range(n_clients):
        idx = np.random.choice(N, size=N // n_clients, replace=False)

        X_client = X[idx].copy()
        avail_mask = np.random.rand(F) < keep_ratio

        X_client[:, ~avail_mask] = np.nan

        client_data.append((X_client, avail_mask.astype(np.float32)))

    return client_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="physionet_fedhf")
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument("--keep-ratio", type=float, default=0.6)
    args = parser.parse_args()

    raw_dir = "raw_physionet"
    folder = download_dataset(raw_dir)

    print("Loading PhysioNet...")
    data = load_physionet_data(folder)
    X = data.values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Federated split
    clients = federated_split(
        X_train,
        n_clients=args.n_clients,
        keep_ratio=args.keep_ratio,
    )

    out_dir = Path(args.out_dir)
    (out_dir / "clients").mkdir(parents=True, exist_ok=True)
    for cid, (Xc, avail) in enumerate(clients):
        np.savez(
            out_dir / "clients" / f"client_{cid}.npz",
            X=Xc,
            availability_mask=avail,
        )

    np.savez(out_dir / "test.npz", X=X_test)

    print(f"\n[OK] PhysioNet FedHF dataset created at: {out_dir}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")


if __name__ == "__main__":
    main()
