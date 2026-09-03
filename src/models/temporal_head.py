"""Spatiotemporal sequence modeling with Bidirectional GRU and Temporal Self-Attention."""

import torch
import torch.nn as nn


class BiGRUTemporalDetector(nn.Module):
    """
    Spatiotemporal video head on frozen dual-stream sequence embeddings.
    Uses 2-layer Bidirectional GRU and temporal self-attention to detect
    single-frame synthesis anomalies and inter-frame flickering.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  Sequence embeddings [Batch, T_frames, embed_dim]
        Output: (video_logits [Batch, 1], frame_attention_weights [Batch, T_frames])
        """
        gru_out, _ = self.gru(x)  # [Batch, T, hidden_dim * 2]
        attn_scores = self.attention(gru_out)  # [Batch, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [Batch, T, 1]
        context = torch.sum(gru_out * attn_weights, dim=1)  # [Batch, hidden_dim * 2]
        video_logit = self.classifier(context)  # [Batch, 1]
        return video_logit, attn_weights.squeeze(-1)
