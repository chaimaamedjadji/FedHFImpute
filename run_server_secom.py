from __future__ import annotations
import argparse
import os
import pandas as pd
import flwr as fl

from fedhf.model import FedHFImputer
from fedhf.utils import ndarrays_from_state_dict


def weighted_avg(metrics):
    total, s = 0, 0.0
    for n, m in metrics:
        if m and "train_loss" in m:
            s += n * float(m["train_loss"])
            total += n
    return {"train_loss": s / max(total, 1)}


def corruption_schedule(r: int) -> float:
    # Hard early = better learning signal
    if r <= 5:
        return 0.8
    if r <= 12:
        return 0.6
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--min-clients", type=int, default=2)

    parser.add_argument("--data-dir", type=str, default="secom_fedhf")
    parser.add_argument("--n-features", type=int, default=591)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    model = FedHFImputer(n_features=args.n_features, d_model=args.d_model, n_layers=args.n_layers)
    initial_params = fl.common.ndarrays_to_parameters(ndarrays_from_state_dict(model))

    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "lr": args.lr,
            "local_epochs": args.local_epochs,
            "block_frac": 0.25,
            "eval_corruption_prob": 0.6,
        }

    def eval_config(server_round: int):
        return {
            "server_round": server_round,
            "corruption_prob": 0.6,
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_params,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=weighted_avg,
    )

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # ---- Save CSV ----
    rows = []
    # distributed loss = evaluation loss (we return RMSE as "loss" on clients)
    if history and history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            rows.append({"round": rnd, "val_rmse": float(loss)})

    df = pd.DataFrame(rows).sort_values("round") if rows else pd.DataFrame({"round": [], "val_rmse": []})

    # fit metrics
    if history and history.metrics_distributed_fit:
        # history.metrics_distributed_fit: Dict[str, List[Tuple[int, Scalar]]]
        if "train_loss" in history.metrics_distributed_fit:
            m = dict(history.metrics_distributed_fit["train_loss"])
            df["train_loss"] = df["round"].map(lambda r: float(m.get(r, float("nan"))))

    out_path = os.path.join("results", "server_history.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
