from typing import Optional
import torch
import torch.nn as nn

class TemporalSequenceEncoder(nn.Module):
    """
    2-Layer Pre-LN Temporal Transformer Encoder for modeling inter-frame dependencies,
    eye-blinking glitches, and boundary temporal flickering across video frames [B, T, D].
    """
    def __init__(
        self,
        embed_dim: int = 1152,
        max_len: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input: x of shape [B, T, D] where T <= max_len
        padding_mask: optional bool tensor [B, T], True = padded position (to be ignored).
        Output: Pooled sequence feature vector [B, D] using masked mean (ignores padding).
        """
        B, T, D = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        pos = self.pos_embed[:, :T, :]
        x_pos = x + pos
        out = self.transformer(x_pos, src_key_padding_mask=padding_mask)

        # Masked mean pooling — ignore padded positions
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1], 1 = valid
            out = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            out = out.mean(dim=1)
        return out
