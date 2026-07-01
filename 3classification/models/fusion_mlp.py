"""
MLP feature fusion module: project image and text features to shared space, then concat and classify
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPFeatureFusion(nn.Module):
    """
    img_feat [B, img_dim] + text_feat [B, text_dim]
    -> project each to projection_dim -> concat -> MLP -> classification
    """

    def __init__(
        self,
        img_dim: int = 256,
        text_dim: int = 768,
        projection_dim: int = 512,
        num_classes: int = 6,
    ):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fusion_backbone = nn.Sequential(
            nn.Linear(projection_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        p_img = self.img_proj(img_feat)  # [B, projection_dim]
        p_text = self.text_proj(text_feat)  # [B, projection_dim]
        combined = torch.cat([p_img, p_text], dim=1)  # [B, projection_dim*2]
        x = self.fusion_backbone(combined)  # [B, 128]
        return self.classifier(x)  # [B, num_classes]
