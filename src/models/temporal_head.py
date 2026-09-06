"""Spatiotemporal sequence modeling with Bidirectional GRU and Temporal Self-Attention."""

import torch
import torch.nn as nn


class BiGRUTemporalDetector(nn.Module):
    """
    Spatiotemporal video head on frozen dual-stream sequence embeddings.
    Uses 2-layer Bidirectional GRU with first-order velocity deltas and
    temporal self-attention to detect spatial and inter-frame synthesis anomalies.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_deltas: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_deltas = use_deltas
        input_size = embed_dim * 2 if use_deltas else embed_dim

        self.gru = nn.GRU(
            input_size=input_size,
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
        if self.use_deltas:
            # First-order temporal velocity: delta_t = x_t - x_{t-1}, delta_0 = 0
            delta = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, 1:, :] - x[:, :-1, :]], dim=1)
            x_in = torch.cat([x, delta], dim=-1)  # [Batch, T, embed_dim * 2]
        else:
            x_in = x

        gru_out, _ = self.gru(x_in)  # [Batch, T, hidden_dim * 2]
        attn_scores = self.attention(gru_out)  # [Batch, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [Batch, T, 1]
        context = torch.sum(gru_out * attn_weights, dim=1)  # [Batch, hidden_dim * 2]
        video_logit = self.classifier(context)  # [Batch, 1]
        return video_logit, attn_weights.squeeze(-1)
