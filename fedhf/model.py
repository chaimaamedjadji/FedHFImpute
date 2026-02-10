from __future__ import annotations
import torch
import torch.nn as nn
from .gnn import GraphSAGELayer, GraphSAGEWeighted

class FedHFImputer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.feature_emb = nn.Embedding(n_features, d_model)
        self.in_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.gnn = nn.ModuleList([GraphSAGEWeighted(d_model, dropout=dropout) for _ in range(n_layers)])

        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),  # numeric imputation
        )

    def forward(self, x_val: torch.Tensor, obs_mask: torch.Tensor, avail_mask: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """
        x_val: (B, F), filled (NaN -> 0)
        obs_mask: (B, F) 1 if observed value exists
        avail_mask: (B, F) 1 if feature exists on this client
        edge_index: (2, E) edges over F nodes
        """
        B, F = x_val.shape
        assert F == self.n_features

        token_num = torch.stack([x_val, obs_mask, avail_mask], dim=-1)  # (B,F,3)
        h = self.in_proj(token_num)  # (B,F,d)

        feat_ids = torch.arange(F, device=x_val.device).unsqueeze(0).expand(B, F)
        h = h + self.feature_emb(feat_ids)

        # Node active mask = avail_mask (unavailable features should not participate)
        node_mask = avail_mask

        for layer in self.gnn:
            #h = h + layer(h, edge_index=edge_index, node_mask=node_mask)
            h = h + layer(h, edge_index=edge_index, edge_weight=edge_weight, node_mask=node_mask)

        y = self.out_head(h).squeeze(-1)  # (B,F)
        return y
