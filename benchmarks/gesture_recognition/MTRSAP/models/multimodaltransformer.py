import math

import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.models as models
import torchaudio.transforms as transforms

from transformers import Wav2Vec2Processor, Wav2Vec2Model



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LeanTemporalEncoder(nn.Module):
    def __init__(self, in_ch, hid=256, k=9, dilations=(1, 2, 4, 8), groups=32):
        super().__init__()
        self.reduce = nn.Conv1d(in_ch, hid, kernel_size=1)
        
        # Adjust groups if hid is not divisible
        if hid % groups != 0:
            groups = min(groups, hid)
            while hid % groups != 0:
                groups -= 1
        
        layers = []
        for d in dilations:
            pad = (k // 2) * d
            layers += [
                nn.Conv1d(hid, hid, kernel_size=k, padding=pad, dilation=d),
                nn.GroupNorm(groups, hid),
                nn.ReLU(inplace=True),
            ]
        self.temporal = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C, T)
        x = self.reduce(x)  # (B, hid, T)
        x = self.temporal(x)  # (B, hid, T)
        return x


class ModularCrossModalAttention(nn.Module):
    """
    Flexible cross-modal attention that works with any number of modalities.
    Each modality attends to all other modalities.
    """
    def __init__(self, d_model, nhead, num_layers, modality_names, dropout=0.1):
        super().__init__()
        
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        
        # Create separate encoders for each modality
        self.modality_encoders = nn.ModuleDict()
        for name in modality_names:
            self.modality_encoders[name] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
                num_layers=num_layers // 2
            )
        
        # Create cross-attention layers: each modality can attend to all others
        self.cross_attentions = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        for src_modality in modality_names:
            for tgt_modality in modality_names:
                if src_modality != tgt_modality:
                    key = f"{src_modality}_to_{tgt_modality}"
                    self.cross_attentions[key] = nn.MultiheadAttention(
                        d_model, nhead, dropout=dropout, batch_first=True
                    )
            self.norms[src_modality] = nn.LayerNorm(d_model)
        
        # Fusion transformer
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers // 2
        )
        
    def forward(self, modality_features):
        """
        Args:
            modality_features: Dict of {modality_name: (B, T_i, d_model)}
        Returns:
            fused: (B, sum(T_i), d_model)
        """
        # Step 1: Encode each modality separately
        encoded_features = {}
        for name, feat in modality_features.items():
            encoded_features[name] = self.modality_encoders[name](feat)
        
        # Step 2: Cross-modal attention - each modality attends to all others
        fused_features = {}
        
        for src_modality in modality_features.keys():
            attended_list = []
            
            for tgt_modality in modality_features.keys():
                if src_modality != tgt_modality:
                    key = f"{src_modality}_to_{tgt_modality}"
                    attended, _ = self.cross_attentions[key](
                        query=encoded_features[src_modality],
                        key=encoded_features[tgt_modality],
                        value=encoded_features[tgt_modality]
                    )
                    attended_list.append(attended)
            
            # Average attention from all other modalities
            if attended_list:
                attended_combined = torch.stack(attended_list, dim=0).mean(dim=0)
                fused_features[src_modality] = self.norms[src_modality](
                    encoded_features[src_modality] + attended_combined
                )
            else:
                # Only one modality present
                fused_features[src_modality] = encoded_features[src_modality]
        
        # Step 3: Concatenate all modalities
        combined = torch.cat([fused_features[name] for name in modality_features.keys()], dim=1)
        
        # Step 4: Final fusion
        output = self.fusion_encoder(combined)
        
        return output


class ModularTemporalEncoder(nn.Module):
    """
    Creates temporal encoders for each modality with configurable parameters.
    """
    def __init__(self, d_model, modality_configs):
        super().__init__()
        
        self.encoders = nn.ModuleDict()
        
        for name, config in modality_configs.items():
            in_ch = config.get('feature_size', d_model)
            k = config.get('kernel_size', 9)
            dilations = config.get('dilations', (1, 2, 4, 8))
            groups = config.get('groups', 32)
            
            self.encoders[name] = LeanTemporalEncoder(
                in_ch=in_ch,
                hid=d_model,
                k=k,
                dilations=dilations,
                groups=groups
            )
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: Dict of {modality_name: (B, T_i, C_i)}
        Returns:
            encoded: Dict of {modality_name: (B, T_i, d_model)}
        """
        encoded = {}
        for name, feat in modality_features.items():
            # Transpose for Conv1d: (B, T, C) -> (B, C, T)
            encoded[name] = self.encoders[name](feat.transpose(1, 2))
            # Transpose back: (B, d_model, T) -> (B, T, d_model)
            encoded[name] = encoded[name].transpose(1, 2)
        
        return encoded


class ModularMultimodalTransformer(nn.Module):
    """
    Fully modular multimodal transformer that handles any combination of modalities.
    
    Usage:
        modality_config = {
            'video': {'feature_size': 2048, 'kernel_size': 9, 'dilations': (1,2,4,8,16)},
            'imu': {'feature_size': 3, 'kernel_size': 5, 'dilations': (1,2,4)},
            'audio': {'feature_size': 768, 'kernel_size': 7, 'dilations': (1,2,4,8)}
        }
    """
    def __init__(self, args, modality_config: dict):
        super().__init__()

        # Model parameters
        self.d_model = args.transformer_params["d_model"]
        self.dropout = args.transformer_params["dropout"]
        self.num_layers = args.transformer_params["num_layers"]
        self.nhead = args.transformer_params["nhead"]
        self.num_classes = args.transformer_params["output_dim"]
        
        self.modality_config = modality_config
        self.modality_names = list(modality_config.keys())
        
        # Feature extractors (optional - only if modality requires it)
        self.feature_extractors = nn.ModuleDict()
        
        # For video modality with ResNet
        if 'video' in modality_config:
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.backbone_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.feature_extractors['video'] = self.backbone
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for name, config in modality_config.items():
            feature_size = config['feature_size']
            self.projections[name] = nn.Linear(feature_size, self.d_model)
        
        # Positional encodings for each modality
        self.positional_encodings = nn.ModuleDict()
        for name in self.modality_names:
            self.positional_encodings[name] = PositionalEncoding(
                d_model=self.d_model, 
                max_len=1000, 
                dropout=self.dropout
            )
        
        # Modality-specific temporal encoders
        self.temporal_encoders = ModularTemporalEncoder(
            d_model=self.d_model,
            modality_configs=modality_config
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = ModularCrossModalAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            modality_names=self.modality_names,
            dropout=self.dropout
        )
        
        # Global pooling for sequence-to-one
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def extract_video_features(self, video):
        """
        Extract ResNet-50 features from video frames.
        Args:
            video: (B, T, C, H, W) - batch of video sequences
        Returns:
            features: (B, T, 2048)
        """
        batch_size, num_frames, c, h, w = video.size()
        video = video.float()
        
        # Reshape to process each frame individually
        video = video.view(batch_size * num_frames, c, h, w)
        
        # Extract features using ResNet backbone
        with torch.no_grad():
            features = self.feature_extractors['video'](video)  # (B*T, 2048, 7, 7)
        
        features = self.backbone_avgpool(features)  # (B*T, 2048, 1, 1)
        features = features.view(batch_size, num_frames, -1)  # (B, T, 2048)
        
        return features
    
    def forward(self, batch):
        """
        Args:
            batch: Dictionary with modality names as keys
                Examples:
                - {'video': (B, T_v, C, H, W)}
                - {'video': (B, T_v, C, H, W), 'imu': (B, T_i, 3)}
                - {'video': ..., 'imu': ..., 'audio': (B, T_a, 768)}
        Returns:
            logits: (B, num_classes) - classification logits
        """
        processed_features = {}
        
        # Process each available modality
        for name in batch.keys():
            if name not in self.modality_names:
                raise ValueError(f"Unknown modality: {name}. Expected one of {self.modality_names}")
            
            # Extract features if needed (e.g., video)
            if name == 'images':
                feat = self.extract_video_features(batch[name])
            else:
                feat = batch[name]
                
            
            # Project to d_model
            feat = self.projections[name](feat)  # (B, T_i, d_model)
            
            # Add positional encoding
            feat = self.positional_encodings[name](feat)  # (B, T_i, d_model)
            
            processed_features[name] = feat
        
        # Modality-specific temporal encoding
        encoded_features = self.temporal_encoders(processed_features)
        
        # Cross-modal fusion (works with any subset of modalities)
        fused = self.cross_modal_fusion(encoded_features)  # (B, sum(T_i), d_model)
        
        # Global pooling: (B, T, d_model) -> (B, d_model)
        fused = fused.transpose(1, 2)  # (B, d_model, T)
        pooled = self.global_pool(fused).squeeze(-1)  # (B, d_model)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits

def build_default_multitransformer(cfg):
    modality_config = cfg.get('modality_config', {
        'video': {'feature_size': 2048, 'kernel_size': 9, 'dilations': (1,2,4,8,16)},
        'imu': {'feature_size': 3, 'kernel_size': 5, 'dilations': (1,2,4)},
        'audio': {'feature_size': 768, 'kernel_size': 7, 'dilations': (1,2,4,8)}
    })
    modal_config = cfg.get('transformer_params', {
        "d_model": 256,
        "dropout": 0.1,
        "num_layers": 6,
        "nhead": 8,
        "output_dim": 10  # Number of gesture classes
    })
    model = ModularMultimodalTransformer(modal_config, modality_config)
    return model

# Example usage:
if __name__ == "__main__":
    # Create mock args
    # test build_default_multitransformer
    cfg = {
        'modality_config': {
            'video': {'feature_size': 2048, 'kernel_size': 9, 'dilations': (1,2,4,8,16)},
            'imu': {'feature_size': 3, 'kernel_size': 5, 'dilations': (1,2,4)},
            'audio': {'feature_size': 768, 'kernel_size': 7, 'dilations': (1,2,4,8)}
        },
        'transformer_params': {
            "d_model": 256,
            "dropout": 0.1,
            "num_layers": 6,
            "nhead": 8,
            "output_dim": 10  # Number of gesture classes
        }
    }
    model = build_default_multitransformer(cfg).cuda()
    print(model)
    # class Args:
    #     transformer_params = {
    #         "d_model": 256,
    #         "dropout": 0.1,
    #         "num_layers": 6,
    #         "nhead": 8,
    #         "output_dim": 10  # Number of gesture classes
    #     }
    
    # args = Args()
    
    # # Define modality configuration
    # modality_config = {
    #     'video': {
    #         'feature_size': 2048,  # ResNet-50 output
    #         'kernel_size': 9,
    #         'dilations': (1, 2, 4, 8, 16),
    #         'groups': 32
    #     },
    #     'imu': {
    #         'feature_size': 3,  # 3D accelerometer
    #         'kernel_size': 5,
    #         'dilations': (1, 2, 4),
    #         'groups': 8
    #     },
    #     'audio': {
    #         'feature_size': 768,  # e.g., wav2vec features
    #         'kernel_size': 7,
    #         'dilations': (1, 2, 4, 8),
    #         'groups': 16
    #     }
    # }
    
    # # Create model with all modalities
    # model = ModularMultimodalTransformer(args, modality_config).cuda()
    
    # print("=" * 50)
    # print("Test 1: All modalities")
    # batch_full = {
    #     'video': torch.randn(2, 16, 3, 224, 224).cuda(),
    #     'imu': torch.randn(2, 100, 3).cuda(),
    #     'audio': torch.randn(2, 80, 768).cuda()
    # }
    # output = model(batch_full)
    # print(f"Output shape: {output.shape}")  # (2, 10)
    
    # print("\n" + "=" * 50)
    # print("Test 2: Only video + imu")
    # batch_two = {
    #     'video': torch.randn(2, 16, 3, 224, 224).cuda(),
    #     'imu': torch.randn(2, 100, 3).cuda()
    # }
    # output = model(batch_two)
    # print(f"Output shape: {output.shape}")  # (2, 10)
    
    # print("\n" + "=" * 50)
    # print("Test 3: Only video")
    # batch_one = {
    #     'video': torch.randn(2, 16, 3, 224, 224).cuda()
    # }
    # output = model(batch_one)
    # print(f"Output shape: {output.shape}")  # (2, 10)
    
    # print("\n" + "=" * 50)
    # print("Model is fully modular and handles any subset of modalities!")