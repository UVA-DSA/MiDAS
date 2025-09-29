# multimodal_modular.py
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.config import ModelCfg

# ---------- Utilities ----------
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B,T,D]
        T = x.shape[1]
        return self.dropout(x + self.pe[:T].unsqueeze(0))

class AttnPool1D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
    def forward(self, x, mask: Optional[torch.Tensor] = None):  # x [B,T,D], mask [B,T]=True for valid
        w = self.score(x).squeeze(-1) / math.sqrt(x.size(-1))  # [B,T]
        if mask is not None:
            w = w.masked_fill(~mask, -1e9)
        w = w.softmax(dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)  # [B,D]

class MLP1D(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, d_model),
        )
    def forward(self, x):  # [B,T,F]
        return self.net(x)


# ---------- Fusion strategies ----------
class ConcatTokensFusion(nn.Module):
    """Concatenate modality token sequences along time and use a shared encoder."""
    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # tokens: {mod: [B,T,D]}
        zs = [z for z in tokens.values() if z is not None]
        return torch.cat(zs, dim=1) if len(zs) > 1 else zs[0]

class LateGatedFusion(nn.Module):
    """Encode each modality separately, pool per-mod, then gate+sum."""
    def __init__(self, d_model, num_modalities):
        super().__init__()
        self.pool = AttnPool1D(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.num_modalities = num_modalities
    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = []
        gates  = []
        for z in tokens.values():
            p = self.pool(z)                 # [B,D]
            g = torch.sigmoid(self.gate(p))  # [B,1]
            pooled.append(p); gates.append(g)
        P = torch.stack(pooled, dim=1)       # [B,M,D]
        G = torch.stack(gates,  dim=1)       # [B,M,1]
        return torch.sum(P * G, dim=1)       # [B,D]

class CrossAttendFusion(nn.Module):
    """Cross-attend from a query modality (e.g., trakstar) to others."""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
    def forward(self, tokens: Dict[str, torch.Tensor], query_mod="trakstar") -> torch.Tensor:
        q = tokens[query_mod]                    # [B,Tq,D]
        others = [v for k, v in tokens.items() if k != query_mod and v is not None]
        if not others:
            return q
        kv = torch.cat(others, dim=1)           # [B, sumTk, D]
        out, _ = self.mha(q, kv, kv)
        return out                               # [B,Tq,D]

# ---------- Main model ----------
class ModularMultimodalTransformer(nn.Module):
    """
    Inputs are a dict of tensors with keys matching cfg.include_modalities.
    Each tensor is [B,T,F_mod]. You can pass any subset at train/eval time.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        # Per-modality encoders (project to d_model)
        self.encoders = nn.ModuleDict()
        for m in cfg.modalities:
            self.encoders[m] = MLP1D(cfg.modalities[m].in_dim, cfg.d_model)

        # Shared transformer encoder (for token-level fusion strategies)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=cfg.num_layers)

        # Type embeddings to tell the transformer which modality a token came from
        self.type_embed = nn.Embedding(num_embeddings=len(cfg.modalities)+1, embedding_dim=cfg.d_model)
        self.mod2id = {m:i+1 for i,m in enumerate(cfg.modalities)}  # 0 reserved
        self.pe = PositionalEncodingBF(cfg.d_model, dropout=cfg.dropout)

        # Fusion head choice
        if cfg.fusion == "concat_tokens":
            self.fuser = ConcatTokensFusion()
            self.pool = AttnPool1D(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.num_classes)
            self.use_transformer = True
        elif cfg.fusion == "late_gated":
            self.fuser = LateGatedFusion(cfg.d_model, num_modalities=len(cfg.include_modalities))
            self.head = nn.Linear(cfg.d_model, cfg.num_classes)
            self.use_transformer = False
        elif cfg.fusion == "cross_attend":
            self.fuser = CrossAttendFusion(cfg.d_model, cfg.nhead, cfg.dropout)
            self.pool = AttnPool1D(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.num_classes)
            self.use_transformer = True
        else:
            raise ValueError(f"Unknown fusion: {cfg.fusion}")

    def _maybe_drop(self, z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if z is None: return None
        if self.training and self.cfg.modality_dropout_p > 0:
            if torch.rand(1).item() < self.cfg.modality_dropout_p:
                return torch.zeros_like(z)
        return z

    def forward(self, inputs: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None):
        """
        inputs: dict {modality: [B,T,F_mod]}
        masks:  dict {modality: [B,T]=True for valid} (optional)
        You may pass ANY subset of modalities listed in cfg.include_modalities.
        """
        tokens: Dict[str, torch.Tensor] = {}
        valids = 0
        for m in self.cfg.include_modalities:
            x = inputs.get(m, None)
            if x is None:  # missing modality
                continue
            z = self.encoders[m](x)  # [B,T,d]
            # add type embedding
            type_id = torch.tensor(self.mod2id[m], device=z.device, dtype=torch.long)
            z = z + self.type_embed(type_id).view(1,1,-1)
            z = self._maybe_drop(z)
            tokens[m] = z
            valids += 1

        if valids == 0:
            raise ValueError("No active modalities provided in forward().")

        if self.cfg.fusion == "late_gated":
            # late fusion works on pooled per-mod representations; no transformer needed
            fused = self.fuser(tokens)               # [B,D]
            logits = self.head(fused)
            return logits

        if self.cfg.fusion == "cross_attend":
            # returns token seq (query modality length)
            fused_tokens = self.fuser(tokens)        # [B,Tq,D]
            z = self.pe(fused_tokens)
            z = self.transformer(z)                  # [B,Tq,D]
            y = self.pool(z, None)                   # [B,D]
            logits = self.head(y)
            return logits

        # concat_tokens
        z = self.fuser(tokens)                       # [B, sum(Tm), D]
        z = self.pe(z)
        z = self.transformer(z)                      # [B, sum(Tm), D]
        y = self.pool(z, None)                       # [B,D]
        logits = self.head(y)
        return logits

# test model
if __name__ == "__main__":
    # Example cfg
    modalities = {
        "trakstar": ModalityCfg(name="trakstar", in_dim=15),
        "watch":    ModalityCfg(name="watch",    in_dim=6),
        "console":  ModalityCfg(name="console",  in_dim=3),
        "imu":      ModalityCfg(name="imu",      in_dim=6),
        "raven":      ModalityCfg(name="raven",      in_dim=12),
    }
    cfg = ModelCfg(
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        num_classes=10,
        include_modalities=["trakstar","watch","console","imu","raven"],
        fusion="concat_tokens",  # "concat_tokens", "late_gated", "cross_attend"
        modality_dropout_p=0.1,
        modalities=modalities
    )
    model = ModularMultimodalTransformer(cfg)
    # print(model)

    B = 2
    window_length = 30 # 1 second at 30Hz


    x_trakstar = torch.randn(B, window_length, modalities["trakstar"].in_dim)
    x_watch    = torch.randn(B, window_length, modalities["watch"].in_dim)
    x_console  = torch.randn(B, window_length, modalities["console"].in_dim)
    x_imu      = torch.randn(B, window_length, modalities["imu"].in_dim)
    x_raven    = torch.randn(B, window_length, modalities["raven"].in_dim)

    inputs = {
        "trakstar": x_trakstar,
        "watch": x_watch,
        "console": x_console,
        "imu": x_imu,
        "raven": x_raven,
    }

    logits = model(inputs)  # [B,num_classes]
    print("logits:", logits.shape)