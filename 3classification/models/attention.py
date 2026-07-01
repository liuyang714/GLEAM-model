"""
Gated Attention modules
  - InstanceGatedAttn : instance-level (patch-level) attention
  - InterglomerularGatedAttn  : interglomerular-level (image-level) attention
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceGatedAttn(nn.Module):
    """Instance-level gated attention: weighted aggregation of patch features for a single image"""

    def __init__(self, in_dim: int = 768, att_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, att_dim)
        self.U = nn.Linear(in_dim, att_dim)
        self.w = nn.Linear(att_dim, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : [N, in_dim] — N patch features of a single image

        Returns
        -------
        M : [in_dim] — aggregated feature vector
        alpha : [N] — attention weights
        """
        hv = torch.tanh(self.V(H))
        hu = torch.sigmoid(self.U(H))
        A = self.w(hv * hu).squeeze(-1)
        alpha = F.softmax(A, dim=0)
        M = (alpha.unsqueeze(-1) * H).sum(dim=0)
        return M, alpha


class InterglomerularGatedAttn(nn.Module):
    """interglomerular-level gated attention: weighted aggregation of multiple image features"""

    def __init__(self, in_dim: int = 768, att_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, att_dim)
        self.U = nn.Linear(in_dim, att_dim)
        self.w = nn.Linear(att_dim, 1)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : [N, in_dim] — aggregated features from N images of a patient

        Returns
        -------
        M : [in_dim] — interglomerular-level feature vector
        alpha : [N] — attention weights
        """
        hv = torch.tanh(self.V(H))
        hu = torch.sigmoid(self.U(H))
        A = self.w(hv * hu).squeeze(-1)
        alpha = F.softmax(A, dim=0)
        M = (alpha.unsqueeze(-1) * H).sum(dim=0)
        return M, alpha
