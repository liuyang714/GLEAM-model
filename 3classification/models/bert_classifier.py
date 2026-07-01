"""
BERT text classifier (with age/sex feature fusion)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT + age/sex → classification

    forward returns logits [B, num_labels];
    return_feat=True returns fused feature vector [B, hidden]
    """

    def __init__(
        self,
        pretrained: str,
        num_labels: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        hidden = self.bert.config.hidden_size

        self.age_fc = nn.Linear(1, 1)
        self.sex_fc = nn.Linear(1, 1)
        self.fusion = nn.Linear(hidden + 2, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
        return_feat: bool = False,
    ) -> torch.Tensor:
        # BERT encoding
        bert_feat = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # [B, H]

        # age/sex features
        age_f = self.relu(self.age_fc(age.unsqueeze(1) if age.dim() == 1 else age))
        sex_f = self.relu(self.sex_fc(sex.unsqueeze(1) if sex.dim() == 1 else sex))

        # Fusion
        combined = torch.cat([bert_feat, age_f, sex_f], dim=1)  # [B, H+2]
        fused = self.relu(self.fusion(combined))  # [B, H]

        if return_feat:
            return fused

        x = self.dropout(fused)
        return self.classifier(x)  # [B, num_labels]
