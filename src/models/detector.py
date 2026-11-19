"""
Dual-Stream Spatial-Frequency Baseline Deepfake Detector (Kumar et al., 2026)
"""
import torch
import torch.nn as nn
from .branches import SpatialBranch, FrequencyBranch
from .cbam import CBAM


class HybridDetector(nn.Module):
    """Dual-stream spatial-frequency deepfake detector with CBAM attention.
    
    Reference Architecture (Kumar et al., 2026):
        Input Image (x)
          ├── Spatial Stream: x -> ResNet-50 -> f_s (2048-d)
          └── Frequency Stream: DCT(x) -> ResNet-50 -> f_f (2048-d)
                  ↓
          Static Concatenation: [f_s, f_f] (4096-d)
                  ↓
          CBAM Channel Attention Module
                  ↓
          Multi-Layer Perceptron (MLP) Classifier Head -> Sigmoid -> P(Fake)

    Ablation modes:
        - 'full': Complete dual-stream architecture with CBAM
        - 'spatial_only': Unimodal spatial stream alone
        - 'frequency_only': Unimodal frequency stream alone
        - 'fusion_no_attention': Dual-stream static concatenation without CBAM
    """
    def __init__(self, mode='full', feature_dim=2048, dropout_rate=0.5, pretrained=True):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim

        if mode in ('full', 'fusion_no_attention', 'spatial_only'):
            self.spatial_branch = SpatialBranch(pretrained=pretrained)
        if mode in ('full', 'fusion_no_attention', 'frequency_only'):
            self.frequency_branch = FrequencyBranch(pretrained=pretrained)

        if mode == 'full':
            self.cbam = CBAM(in_features=feature_dim * 2)

        classifier_in = feature_dim if mode in ('spatial_only', 'frequency_only') else feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial_input, frequency_input=None, return_features=False):
        """
        Args:
            spatial_input: RGB image tensor (batch, 3, H, W)
            frequency_input: DCT tensor (batch, 3, H, W)
            return_features: If True, returns intermediate latent representations
        Returns:
            probability: (batch, 1) in [0, 1] — probability of synthetic manipulation
            features_dict: (optional) dictionary of branch representations
        """
        if self.mode == 'spatial_only':
            f_s = self.spatial_branch(spatial_input)
            features = f_s
            f_f = None
        elif self.mode == 'frequency_only':
            assert frequency_input is not None, "Frequency tensor required for frequency_only"
            f_f = self.frequency_branch(frequency_input)
            features = f_f
            f_s = None
        elif self.mode == 'fusion_no_attention':
            assert frequency_input is not None, "Frequency tensor required for dual-stream"
            f_s = self.spatial_branch(spatial_input)
            f_f = self.frequency_branch(frequency_input)
            features = torch.cat([f_s, f_f], dim=1)
        elif self.mode == 'full':
            assert frequency_input is not None, "Frequency tensor required for full dual-stream"
            f_s = self.spatial_branch(spatial_input)
            f_f = self.frequency_branch(frequency_input)
            fused = torch.cat([f_s, f_f], dim=1)
            features = self.cbam(fused)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        prob = self.classifier(features)

        if return_features:
            return prob, {"spatial": f_s, "frequency": f_f, "fused": features}
        return prob
