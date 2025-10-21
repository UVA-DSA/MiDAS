
"""
multitranstcn.py

Multi‑Modal TCN -> Transformer Encoder model.

- One TCN per modality (audio / vision / imu or any custom list)
- Modalities are temporally aligned by (optional) linear interpolation to a common T.
- Fused with a learned per‑modality weighting (softmax over modalities).
- A single Transformer encoder consumes the fused sequence.
- Outputs frame‑wise logits of shape (B, C, T) by default, or sequence logits (B, C) if seq_to_one=True.

This file is self‑contained and does not depend on external config packages.
"""

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field


@dataclass
class MultiMTRSAPModelCfg:
    modalities: List[str] = field(default_factory=list)       # derived
    in_dims: List[int] = field(default_factory=list)          # derived
    num_classes: int = 8
    tcn_channels: List[int] = field(default_factory=lambda: [128, 128, 256])
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    seq_to_one: bool = False
    causal_tcn: bool = True
 

# -----------------------------
# Utilities
# -----------------------------

class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding for [B, T, D]."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [T, D]
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.shape[1]
        return self.dropout(x + self.pe[:T].unsqueeze(0))


def linear_interpolate_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    Linearly resample a [B, T, D] sequence to [B, target_T, D].
    Uses torch.nn.functional.interpolate on a [B, D, T] view.
    """
    if x.size(1) == target_T:
        return x
    # [B, T, D] -> [B, D, T]
    x_ch_first = x.transpose(1, 2)
    x_rs = F.interpolate(x_ch_first, size=target_T, mode='linear', align_corners=False)
    return x_rs.transpose(1, 2).contiguous()


# -----------------------------
# TCN building blocks
# -----------------------------

class TemporalBlock(nn.Module):
    """
    A single TCN block: Dilated causal Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout
    with residual connection. Maintains length (same padding).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        # padding chosen to keep length
        pad = (kernel_size - 1) * dilation if causal else ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.causal = causal
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv1(x)
        if self.causal and self.pad > 0:
            out = out[:, :, :-self.pad]  # remove right padding for strict causality
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        if self.causal and self.pad > 0:
            out = out[:, :, :-self.pad]
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.network(x)


# -----------------------------
# Fusion
# -----------------------------

class LearnedModalAveraging(nn.Module):
    """
    Softmax weight over modalities. Given a list of modality tensors [B, T, D],
    returns a fused tensor [B, T, D] as a weighted average across modalities.
    """
    def __init__(self, num_modalities: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # xs: list of [B, T, D], same T and D
        assert len(xs) > 0
        if len(xs) == 1:
            return xs[0]
        weights = torch.softmax(self.logits, dim=0)  # [M]
        fused = 0.0
        for i, x in enumerate(xs):
            fused = fused + weights[i] * x
        return fused


# -----------------------------
# Multi‑modal TCN -> Transformer
# -----------------------------

class MultiTransTCN(nn.Module):
    """
    Multi‑Modal TCN -> Transformer model.

    Inputs:
      - features: Dict[str, torch.Tensor], where each value is [B, T_m, F_m] for modality m
      - (optional) lengths: Dict[str, torch.Tensor] or None, giving valid lengths per modality [B]

    Config (key args):
      - modalities: list of modality names to expect. (e.g., ["audio", "vision", "imu"])
      - in_dims: dict from modality to feature dim
      - tcn_channels: list of hidden channels for TCN (last element is projected to d_model)
      - d_model: transformer embedding size
      - nhead, num_layers, dim_feedforward, dropout
      - seq_to_one: if True, applies attentive pooling to emit (B, C); else (B, C, T)
      - num_classes: output classes C
      - causal_tcn: use causal conv (default True)
    """
    def __init__(
        self,
        modalities: List[str],
        in_dims: Dict[str, int],
        num_classes: int,
        tcn_channels: List[int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        seq_to_one: bool = False,
        causal_tcn: bool = True,
    ):
        super().__init__()
        assert len(modalities) > 0, "Provide at least one modality"
        self.modalities = modalities
        print("Using modalities:", self.modalities)
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.seq_to_one = seq_to_one

        # one TCN (with projection) per modality ending at d_model
        self.backbones = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for m in modalities:
            in_ch = in_dims[m]
            tcn = TemporalConvNet(
                in_channels=in_ch,
                channels=tcn_channels,
                kernel_size=31,
                dropout=dropout,
                causal=causal_tcn,
            )
            self.backbones[m] = tcn
            self.projections[m] = nn.Conv1d(tcn_channels[-1], d_model, kernel_size=1)

        # fusion: learned average over modalities
        self.fusion = LearnedModalAveraging(num_modalities=len(modalities))

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expects [B, T, D]
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, dropout=dropout, max_len=4096)

        # output heads
        self.out = nn.Linear(d_model, num_classes)
        self.attn_pool = nn.Linear(d_model, 1)  # for seq_to_one

    def _compute_valid_mask(self, lens_by_mod: Dict[str, Optional[torch.Tensor]], T: int, device) -> torch.Tensor:
        """
        If lengths are provided, build a mask [B, T] that is True for valid time steps.
        When multiple modalities, a time step is valid if ANY modality is valid at that time.
        """
        masks = []
        B = None
        for m in self.modalities:
            L = lens_by_mod.get(m, None)
            if L is None:
                return torch.ones(1, T, dtype=torch.bool, device=device)  # default to "all valid" and broadcast
            if B is None:
                B = L.shape[0]
            ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            masks.append(ar < L.unsqueeze(1))
        # OR across modalities
        mask = masks[0]
        for k in range(1, len(masks)):
            mask = mask | masks[k]
        return mask

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        lengths: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        features: dict of modality -> [B, T_m, F_m]
        lengths: dict of modality -> [B] (optional)
        returns:
          - (B, C, T) if seq_to_one == False
          - (B, C) if seq_to_one == True
        """
        assert set(features.keys()) == set(self.modalities), \
            f"Expected modalities {self.modalities}, got {list(features.keys())}"

        # 1) Determine target T and per‑modality backbones
        B = None
        T_target = 0
        per_mod_seq = {}
        for m in self.modalities:
            x_m = features[m]  # [B, T_m, F_m]
            assert x_m.dim() == 3, f"{m} must be [B, T, F]"
            if B is None:
                B = x_m.size(0)
            T_target = max(T_target, x_m.size(1))
            per_mod_seq[m] = x_m

        # 2) Run each modality through its TCN -> project to d_model, resample to common T
        per_mod_repr = []
        for m in self.modalities:
            x = per_mod_seq[m]                        # [B, T_m, F_m]
            x = x.transpose(1, 2).contiguous()        # [B, F_m, T_m]  (channels first for Conv1d)
            x = self.backbones[m](x)                  # [B, tcn_hidden, T_m]
            x = self.projections[m](x)                # [B, d_model, T_m]
            x = x.transpose(1, 2).contiguous()        # [B, T_m, d_model]
            if x.size(1) != T_target:
                x = linear_interpolate_time(x, T_target)  # [B, T_target, d_model]
            per_mod_repr.append(x)

        # 3) Fuse across modalities with learned weights
        x = self.fusion(per_mod_repr)                 # [B, T, d_model]

        # 4) Positional encoding + Transformer encoder
        x = self.posenc(x)                            # [B, T, d_model]
        # Build src_key_padding_mask where True marks PAD positions (invalid)
        if lengths is not None:
            mask_valid = self._compute_valid_mask(lengths, T_target, device=x.device)  # [B, T] (True if valid)
            pad_mask = ~mask_valid                                                        # True indicates padding
            if pad_mask.shape[0] != x.shape[0]:  # broadcast case
                pad_mask = pad_mask.expand(x.shape[0], -1)
        else:
            pad_mask = None
        x = self.transformer(x, src_key_padding_mask=pad_mask)   # [B, T, d_model]

        # 5) Heads
        if self.seq_to_one:
            # attentive pooling for sequence classification -> [B, C]
            scores = self.attn_pool(x).squeeze(-1) / math.sqrt(x.size(-1))  # [B, T]
            if pad_mask is not None:
                scores = scores.masked_fill(pad_mask, float('-inf'))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)                 # [B, T, 1]
            pooled = (x * w).sum(dim=1)                                     # [B, d_model]
            return self.out(pooled)                                         # [B, C]
        else:
            # frame‑wise logits -> [B, C, T]
            logits = self.out(x)                                            # [B, T, C]
            return logits.transpose(1, 2).contiguous()                      # [B, C, T]


# -----------------------------
# Optional convenience wrapper
# -----------------------------

def build_default_multitranstcn(
    cfg: MultiMTRSAPModelCfg
) -> MultiTransTCN:
    """
    Convenience builder with a reasonable default stack:
      - modalities: keys of in_dims
      - TCN channels: [128, 128, 256] -> projected to d_model=256
      - Transformer: 4 layers, 8 heads, FFN=1024, dropout=0.1
    """

    print("Building MultiTransTCN with cfg:", cfg)

    # modalities must be a dict with feature name and dim

    modalities = list(cfg.modalities)

    # replace "images" with "images_feat" in modalities list
    if "images" in modalities:
        modalities[modalities.index("images")] = "images_feat"

    modality_dict = {}
    for i, m in enumerate(modalities):
        if m == "images":
            modality_dict["images_feat"] = cfg.in_dims[i]
        else:
            modality_dict[m] = cfg.in_dims[i]

    print("Modality dict:", modality_dict)

    model = MultiTransTCN(
        modalities=modalities,
        in_dims=modality_dict,
        num_classes=cfg.num_classes,
        tcn_channels=cfg.tcn_channels,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        seq_to_one=cfg.seq_to_one,
        causal_tcn=cfg.causal_tcn,
    )
    return model


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Suppose we have three modalities with different feature dims and lengths
    B = 2
    Tv, Ti = 300, 300
    Fv, Fi = 2048, 14
    C = 19

    x_vision = torch.randn(B, Tv, Fv)
    x_imu = torch.randn(B, Ti, Fi)


    model = build_default_multitranstcn({ "vision": Fv, "imu": Fi}, num_classes=C, seq_to_one=False)

    y = model(
        {"vision": x_vision, "imu": x_imu},
    )
    print("frame‑wise logits:", y.shape)  # [B, C, T]

    # Sequence classification variant
    model2 = build_default_multitranstcn({"vision": Fv, "imu": Fi}, num_classes=C, seq_to_one=True)
    y2 = model2({"vision": x_vision, "imu": x_imu})
    print("sequence logits:", y2.shape)   # [B, C]
