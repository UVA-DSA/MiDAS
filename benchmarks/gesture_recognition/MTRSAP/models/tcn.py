# models/tcn.py
from __future__ import annotations
from typing import Dict, Optional, Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# -------------------------
# utils
# -------------------------

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: [B, T, C]
    mask: [B, T] bool (True = valid). If None, simple mean over T.
    returns: [B, C]
    """
    if mask is None:
      return x.mean(dim=1)
    m = mask.float()  # [B, T]
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom

def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    logits: [..., T]
    mask:   [..., T] bool (True = valid)
    """
    if mask is None:
        return F.softmax(logits, dim=dim)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    return F.softmax(logits, dim=dim)

# -------------------------
# building blocks
# -------------------------

class DepthwiseSeparableConv1d(nn.Module):
    """
    DW separable 1D conv: depthwise T-conv then pointwise. Faster/smaller than full conv on many channels.
    Input: [B, C, T]  Output: [B, C_out, T]
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # 'same' padding for odd kernel
        self.depthwise = weight_norm(nn.Conv1d(c_in, c_in, kernel_size,
                                               padding=padding, dilation=dilation,
                                               groups=c_in, bias=False))
        self.pointwise = weight_norm(nn.Conv1d(c_in, c_out, kernel_size=1, bias=True))
        self.bn = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x

class TemporalBlock(nn.Module):
    """
    Residual block with two depthwise-separable dilated convs.
    Input/Output: [B, C, T]
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(c_in, c_out, kernel_size, dilation, dropout)
        self.conv2 = DepthwiseSeparableConv1d(c_out, c_out, kernel_size, dilation, dropout)
        self.res = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        res = self.res(x)
        return self.act(out + res)

class TemporalEncoder(nn.Module):
    """
    Stacked TemporalBlocks with exponentially increasing dilation.
    Accepts [B, T, F] -> returns [B, T, C]
    """
    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int] = (64, 64, 96, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size for 'same' padding."
        self.in_ln = nn.LayerNorm(input_dim)  # stabilize across features
        c_prev = input_dim
        blocks = []
        for i, c in enumerate(channels):
            dilation = 2 ** i
            blocks.append(TemporalBlock(c_prev, c, kernel_size, dilation, dropout))
            c_prev = c
        self.net = nn.Sequential(*blocks)

    @property
    def output_dim(self) -> int:
        # final channel dim
        last: TemporalBlock = self.net[-1]
        # infer from last conv2 pointwise out_channels
        return last.conv2.pointwise.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        returns: [B, T, C]
        """
        x = self.in_ln(x)
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.net(x)        # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        return x

class AttentionPool(nn.Module):
    """
    Single-head attention pooling over time with mask support.
    Input:  x [B, T, C], mask [B, T] (True = valid)
    Output: pooled [B, C]
    """
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, T, C]
        logits = self.score(x).squeeze(-1)  # [B, T]
        attn = masked_softmax(logits, mask, dim=-1)  # [B, T]
        out = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, C]
        return out

# -------------------------
# Heads
# -------------------------

class ClassificationHead(nn.Module):
    def __init__(self, c_in: int, num_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

# -------------------------
# Models
# -------------------------

class TCNClassifier(nn.Module):
    """
    Single-branch TCN.
      - Input: x [B, T, F]
      - Optional mask: [B, T] bool (True = valid)
      - Output: logits [B, num_classes]
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tcn_channels: Sequence[int] = (64, 64, 96, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        head_hidden: int = 128,
        pool: str = "attention",   # "attention" or "mean"
    ):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, tcn_channels, kernel_size, dropout)
        c = self.encoder.output_dim
        self.pool_type = pool
        self.pool = AttentionPool(c) if pool == "attention" else None
        self.head = ClassificationHead(c, num_classes, hidden=head_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, F]
        mask: [B, T] bool (True = valid)
        """
        h = self.encoder(x)  # [B, T, C]
        if self.pool_type == "attention":
            z = self.pool(h, mask)  # [B, C]
        else:
            z = masked_mean(h, mask)  # [B, C]
        logits = self.head(z)  # [B, K]
        return logits

class MultiBranchTCNClassifier(nn.Module):
    """
    Multi-branch TCN with per-modality encoders, late fusion, and attention pooling.
    Accepts dict inputs that match your dataset collate (any subset of modalities).

      inputs: Dict[str, Tensor] where each Tensor is [B, T, F_m]
      masks:  Optional[Dict[str, Tensor]] with [B, T] bool per modality
              (or a single mask [B, T] applied to all)

      Output: logits [B, num_classes]
    """
    def __init__(
        self,
        modality_input_dims: Dict[str, int],   # e.g., {"trakstar": 12, "console": 8, "sw_left": 3}
        num_classes: int,
        shared_tcn_channels: Sequence[int] = (48, 64, 96),  # keep compact for low data
        kernel_size: int = 5,
        dropout: float = 0.25,
        fusion: str = "concat",               # "concat" or "mean"
        head_hidden: int = 128,
        pool: str = "attention",
    ):
        super().__init__()
        assert len(modality_input_dims) > 0, "Provide at least one modality."

        self.branches = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        self.pool_type = pool
        self.fusion = fusion

        # Build encoders and pools per modality
        out_dims = []
        for m, d in modality_input_dims.items():
            enc = TemporalEncoder(d, shared_tcn_channels, kernel_size, dropout)
            self.branches[m] = enc
            c = enc.output_dim
            out_dims.append(c)
            self.pools[m] = AttentionPool(c) if pool == "attention" else nn.Identity()

        # Fusion dimension
        if fusion == "concat":
            fused_dim = sum(out_dims)
        elif fusion == "mean":
            fused_dim = out_dims[0]
        else:
            raise ValueError("fusion must be 'concat' or 'mean'")

        self.head = ClassificationHead(fused_dim, num_classes, hidden=head_hidden, dropout=dropout)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        inputs: dict of {modality: [B, T, F_m]}
        masks:  either a single [B, T] bool mask, or dict of modality masks
        """
        zs = []
        for m, x in inputs.items():
            mask_m = None
            if masks is not None:
                mask_m = masks[m] if isinstance(masks, dict) else masks
            h = self.branches[m](x)  # [B, T, C]
            if self.pool_type == "attention":
                z = self.pools[m](h, mask_m)  # [B, C]
            else:
                z = masked_mean(h, mask_m)    # [B, C]
            zs.append(z)

        if self.fusion == "concat":
            fused = torch.cat(zs, dim=-1)
        else:  # mean
            # ensure same dim, otherwise use a small projection; for simplicity assume same out_dim
            fused = torch.stack(zs, dim=0).mean(dim=0)

        logits = self.head(fused)  # [B, K]
        return logits

# -------------------------
# Example usage with your dataset batch
# -------------------------

if __name__ == "__main__":
    # Fake example to illustrate shapes
    B, T = 4, 64
    x_trak = torch.randn(B, T, 12)     # e.g., trakstar features
    x_sw   = torch.randn(B, T, 6)      # smartwatch L/R
    x_cons = torch.randn(B, T, 8)      # console
    mask   = torch.ones(B, T, dtype=torch.bool)

    # Single-branch (concatenate features yourself)
    x_all = torch.cat([x_trak, x_sw, x_cons], dim=-1)  # [B, T, F]
    single = TCNClassifier(input_dim=x_all.shape[-1], num_classes=8)
    logits_single = single(x_all, mask)                # [B, 8]

    # Multi-branch (pass a dict)
    mb = MultiBranchTCNClassifier(
        modality_input_dims={"trakstar": 12, "sw": 6, "console": 8},
        num_classes=8,
        shared_tcn_channels=(48, 64, 96),
        kernel_size=5,
        dropout=0.25,
        fusion="concat",
        pool="attention",
    )
    logits_mb = mb({"trakstar": x_trak, "sw": x_sw, "console": x_cons},
                   masks=mask)  # or masks={"trakstar": mask, "sw": mask, "console": mask}
    print(logits_single.shape, logits_mb.shape)
