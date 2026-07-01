"""
GAAN (Gated Attention Attention Network) image classifier
Dual-level attention: intraglomerular-level -> interglomerular-level
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import InstanceGatedAttn, InterglomerularGatedAttn


class GaanClassifier(nn.Module):
    """
    Dual-level GAAN: InstanceAttn -> PatientAttn -> FC -> classification

    forward takes a list (batch of bags), each bag shape [N_i, ...]
    return_feat=True returns hidden-dim feature [B, hidden_dim]
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 6,
        att_dim: int = 128,
    ):
        super().__init__()
        self.inst_attn = InstanceGatedAttn(feature_dim, att_dim)
        self.pat_attn = InterglomerularGatedAttn(feature_dim, att_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: list[torch.Tensor], return_feat: bool = False
    ) -> torch.Tensor:
        batch_feat = []
        for bag in x:
            N = bag.size(0)
            img_feats = []
            for i in range(N):
                f, _ = self.inst_attn(bag[i])
                img_feats.append(f.unsqueeze(0))  # [1, feature_dim]

            img_feats = torch.cat(img_feats, 0)  # [N, feature_dim]
            pat_feat, _ = self.pat_attn(img_feats)  # [feature_dim]
            batch_feat.append(pat_feat.unsqueeze(0))  # [1, feature_dim]

        batch_feat = torch.cat(batch_feat, 0)  # [B, feature_dim]
        h = F.relu(self.fc1(batch_feat))  # [B, hidden_dim]

        if return_feat:
            return h

        return self.fc2(h)  # [B, num_classes]
