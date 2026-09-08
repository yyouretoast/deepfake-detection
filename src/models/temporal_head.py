"""Spatiotemporal sequence modeling with Bidirectional GRU and Temporal Self-Attention."""

import torch
import torch.nn as nn


class BiGRUTemporalDetector(nn.Module):
    """
    Spatiotemporal video head on frozen dual-stream sequence embeddings.
    Uses 2-layer Bidirectional GRU with first-order velocity deltas,
    temporal self-attention, and dual-path extreme-value max pooling
    to detect spatial and inter-frame synthesis anomalies (capturing both
    global temporal consistency and transient 1-frame glitches).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_deltas: bool = True,
        use_max_pool: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_deltas = use_deltas
        self.use_max_pool = use_max_pool
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
        classifier_in = hidden_dim * 4 if use_max_pool else hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Auto-detects whether checkpoint was trained with or without max pooling and deltas."""
        if "classifier.0.weight" in state_dict:
            in_features = state_dict["classifier.0.weight"].shape[1]
            if in_features == self.hidden_dim * 4 and not self.use_max_pool:
                self.use_max_pool = True
                self.classifier[0] = nn.Linear(self.hidden_dim * 4, 128)
            elif in_features == self.hidden_dim * 2 and self.use_max_pool:
                self.use_max_pool = False
                self.classifier[0] = nn.Linear(self.hidden_dim * 2, 128)

        if "gru.weight_ih_l0" in state_dict:
            gru_in = state_dict["gru.weight_ih_l0"].shape[1]
            if gru_in == self.embed_dim and self.use_deltas:
                self.use_deltas = False
                self.gru = nn.GRU(
                    input_size=self.embed_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=self.dropout,
                )
            elif gru_in == self.embed_dim * 2 and not self.use_deltas:
                self.use_deltas = True
                self.gru = nn.GRU(
                    input_size=self.embed_dim * 2,
                    hidden_size=self.hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=self.dropout,
                )

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

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
        attn_context = torch.sum(gru_out * attn_weights, dim=1)  # [Batch, hidden_dim * 2]

        if self.use_max_pool:
            max_context, _ = torch.max(gru_out, dim=1)  # [Batch, hidden_dim * 2]
            context = torch.cat([attn_context, max_context], dim=-1)  # [Batch, hidden_dim * 4]
        else:
            context = attn_context

        video_logit = self.classifier(context)  # [Batch, 1]
        return video_logit, attn_weights.squeeze(-1)
