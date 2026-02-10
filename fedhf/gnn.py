from __future__ import annotations
import torch
import torch.nn as nn

class GraphSAGEWeighted(nn.Module):
    """
    Weighted GraphSAGE:
      h_i <- MLP([h_i || weighted_mean_{j in N(i)} (w_ji * h_j)])

    Supports batch by offsetting node indices.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, node_mask=None):
        """
        h: (B, F, d)
        edge_index: (2, E)
        edge_weight: (E,) in [0,1]
        node_mask: (B, F) optional, 1 active, 0 inactive
        """
        B, F, d = h.shape
        device = h.device

        h_flat = h.reshape(B * F, d)

        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        w = edge_weight      # (E,)

        offsets = torch.arange(B, device=device, dtype=torch.long) * F

        src_all = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_all = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

        w_all = w.unsqueeze(0).expand(B, -1).reshape(-1).to(h_flat.dtype)  # (B*E,)

        agg = torch.zeros_like(h_flat)
        agg.index_add_(0, dst_all, h_flat[src_all] * w_all.unsqueeze(1))

        deg = torch.zeros(B * F, device=device, dtype=h_flat.dtype)
        deg.index_add_(0, dst_all, w_all)
        deg = deg.clamp_min(1e-6).unsqueeze(1)

        agg = agg / deg

        if node_mask is not None:
            m = node_mask.reshape(B * F, 1).to(h_flat.dtype)
            agg = agg * m
            h_flat = h_flat * m

        out = self.mlp(torch.cat([h_flat, agg], dim=1))
        return out.reshape(B, F, d)


class GraphSAGELayer(nn.Module):
    """
    Simple GraphSAGE-like layer:
      h_i <- MLP([h_i || mean_{j in N(i)} h_j])

    Works per-sample, but implemented in batch by offsetting node indices.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, node_mask: torch.Tensor | None = None):
        """
        h: (B, F, d)
        edge_index: (2, E) edges over F feature-nodes (same for all samples)
        node_mask: (B, F) optional mask; 1 for active nodes, 0 for inactive (unavailable)
        """
        B, F, d = h.shape
        E = edge_index.shape[1]
        device = h.device

        # Flatten nodes across batch
        h_flat = h.reshape(B * F, d)

        # Build batch-offset edge indices so edges stay within each sample
        # For each batch b, node ids shift by b*F
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        offsets = torch.arange(B, device=device, dtype=torch.long) * F  # (B,)

        src_all = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)  # (B*E,)
        dst_all = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)  # (B*E,)

        # Aggregate messages: sum neighbor h_src into dst
        agg = torch.zeros_like(h_flat)  # (B*F, d)
        agg.index_add_(0, dst_all, h_flat[src_all])

        # Degree for mean aggregation
        deg = torch.zeros(B * F, device=device, dtype=h_flat.dtype)
        deg.index_add_(0, dst_all, torch.ones_like(dst_all, dtype=h_flat.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)  # (B*F, 1)
        agg = agg / deg

        # Optional mask to zero-out inactive nodes (unavailable features)
        if node_mask is not None:
            m = node_mask.reshape(B * F, 1).to(h_flat.dtype)
            agg = agg * m
            h_flat = h_flat * m

        out = self.mlp(torch.cat([h_flat, agg], dim=1))  # (B*F, d)
        out = out.reshape(B, F, d)
        return out
