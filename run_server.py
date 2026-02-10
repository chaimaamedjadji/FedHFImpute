from __future__ import annotations
import argparse
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


def corruption_schedule(server_round: int) -> float:
    # Start hard, then ease.
    if server_round <= 5:
        return 0.7
    elif server_round <= 12:
        return 0.5
    else:
        return 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--n-features", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)

    # Graph settings (sent as scalars)
    parser.add_argument("--graph-k", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    # Initial parameters
    model = FedHFImputer(n_features=args.n_features, d_model=args.d_model, n_layers=args.n_layers)
    initial_params = fl.common.ndarrays_to_parameters(ndarrays_from_state_dict(model))

    def fit_config(server_round: int):
        return {
            "n_features": int(args.n_features),
            "graph_k": int(args.graph_k),
            "lr": float(args.lr),
            "local_epochs": int(args.local_epochs),
            "corruption_prob": float(corruption_schedule(server_round)),
        }

    def eval_config(server_round: int):
        # Evaluate at fixed difficulty for stable curves
        return {
            "n_features": int(args.n_features),
            "graph_k": int(args.graph_k),
            "corruption_prob": 0.5,
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

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
