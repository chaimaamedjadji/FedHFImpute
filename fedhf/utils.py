from __future__ import annotations
import numpy as np
import torch
from typing import List

def build_structured_graph(n_nodes: int, k: int = 8) -> np.ndarray:
    """
    Deterministic graph: connect each node i to its +/- neighbors.
    k should be even ideally. Edges are undirected represented as two directed edges.

    Returns:
      edge_index: (2, E) int64
    """
    k = max(2, int(k))
    half = k // 2
    edges = set()

    for i in range(n_nodes):
        for d in range(1, half + 1):
            j1 = (i - d) % n_nodes
            j2 = (i + d) % n_nodes
            edges.add((i, j1)); edges.add((j1, i))
            edges.add((i, j2)); edges.add((j2, i))

    edge_index = np.array(list(edges), dtype=np.int64).T
    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return edge_index

def ndarrays_from_state_dict(model: torch.nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def load_state_dict_from_ndarrays(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    sd = model.state_dict()
    keys = list(sd.keys())
    assert len(keys) == len(params), "Parameter length mismatch"
    for k, p in zip(keys, params):
        sd[k] = torch.tensor(p)
    model.load_state_dict(sd, strict=True)
