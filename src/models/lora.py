from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for Conv2d layers (groups=1 only).
    Freezes original weights W_0 and injects trainable low-rank matrices A and B.
    Supports weight folding for 0ms extra inference latency.
    """
    def __init__(
        self,
        base_conv: nn.Conv2d,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05
    ) -> None:
        super().__init__()
        assert base_conv.groups == 1, (
            f"LoRAConv2d only supports groups=1 convolutions. "
            f"Got groups={base_conv.groups}. Use LoRALinear for depthwise/grouped layers."
        )
        self.base_conv = base_conv
        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.kernel_size = base_conv.kernel_size
        self.stride = base_conv.stride
        self.padding = base_conv.padding
        self.dilation = base_conv.dilation
        self.groups = base_conv.groups

        self.base_conv.weight.requires_grad = False
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.merged = False

        if r > 0:
            kw, kh = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            # lora_A: (r, in_channels * kw * kh), lora_B: (out_channels, r)
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_channels * kw * kh))
            self.lora_B = nn.Parameter(torch.zeros(self.out_channels, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base_conv.weight

    @property
    def bias(self):
        return self.base_conv.bias

    def merge_weights(self) -> None:
        """Folds delta W directly into base weights for zero inference latency overhead."""
        if self.r > 0 and not self.merged:
            kw, kh = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            delta_w = (self.lora_B @ self.lora_A).view(
                self.out_channels, self.in_channels, kw, kh
            )
            self.base_conv.weight.data += delta_w * self.scaling
            self.merged = True

    def unmerge_weights(self) -> None:
        """Unfolds delta W from base weights."""
        if self.r > 0 and self.merged:
            kw, kh = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            delta_w = (self.lora_B @ self.lora_A).view(
                self.out_channels, self.in_channels, kw, kh
            )
            self.base_conv.weight.data -= delta_w * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return self.base_conv(x)
        base_out = self.base_conv(x)
        drop_x = self.lora_dropout(x)
        kw, kh = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        delta_w = (self.lora_B @ self.lora_A).view(
            self.out_channels, self.in_channels, kw, kh
        )
        lora_out = F.conv2d(
            drop_x, delta_w, bias=None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        return base_out + lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for nn.Linear layers.
    Standard LoRA as in the original paper: y = W_0 x + (B A) x * scaling
    Works for both MLP projection layers and attention projections.
    Supports weight folding for 0ms extra inference latency.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.merged = False

        if r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            # lora_A: (r, in_features), lora_B: (out_features, r)
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def merge_weights(self) -> None:
        """Folds LoRA delta into base linear weights."""
        if self.r > 0 and not self.merged:
            self.base_linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge_weights(self) -> None:
        """Unfolds LoRA delta from base linear weights."""
        if self.r > 0 and self.merged:
            self.base_linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return self.base_linear(x)
        base_out = self.base_linear(x)
        drop_x = self.lora_dropout(x)
        lora_out = F.linear(drop_x, self.lora_B @ self.lora_A)
        return base_out + lora_out * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_keywords: Tuple[str, ...] = ("fc1", "fc2"),
    skip_grouped_conv: bool = True,
) -> int:
    """
    Freezes all backbone parameters and replaces target layers with LoRA wrappers.

    Default targets 'fc1' and 'fc2' (the MLP projection Linear layers in ConvNeXt-Base).
    For Conv2d layers with groups=1, uses LoRAConv2d.
    For nn.Linear layers, uses LoRALinear.
    Grouped/depthwise Conv2d layers are skipped by default (skip_grouped_conv=True)
    because their parameter count is negligible and the matmul shapes are incompatible.
    """
    # Step 1: Freeze all base parameters
    for p in model.parameters():
        p.requires_grad = False

    lora_count = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not any(kw in child_name for kw in target_keywords):
                continue
            if isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, r=rank, lora_alpha=alpha)
                setattr(module, child_name, lora_layer)
                lora_count += 1
            elif isinstance(child, nn.Conv2d):
                if skip_grouped_conv and child.groups > 1:
                    continue
                lora_layer = LoRAConv2d(child, r=rank, lora_alpha=alpha)
                setattr(module, child_name, lora_layer)
                lora_count += 1

    return lora_count


def merge_all_lora_weights(model: nn.Module) -> None:
    """Folds all LoRA weights into base parameters across model."""
    for module in model.modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            module.merge_weights()


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Returns state_dict containing ONLY LoRA parameters and classification head."""
    return {
        k: v.cpu() for k, v in model.state_dict().items()
        if "lora_" in k or "classifier" in k or "cross_attn" in k
        or "spatial_proj" in k or "freq_proj" in k
    }
