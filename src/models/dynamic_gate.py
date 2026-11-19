"""
Dynamic Gated Fusion Architecture (Singh & Sagar, 2026)
"""
import torch
import torch.nn as nn
from .branches import SpatialBranch, FrequencyBranch


class DynamicGatedFusion(nn.Module):
    """Dynamic Gated Fusion Module.
    
    Evaluates both spatial and frequency latent streams and predicts an
    adaptive gating weight alpha in [0, 1]:
        alpha -> 1.0 : Rely predominantly on the Spatial stream (e.g. under Resizing)
        alpha -> 0.3 : Rely predominantly on the Frequency stream (e.g. under JPEG)
    
    Mathematical Formulation:
        alpha = Sigmoid( MLP( [f_s, f_f] ) )
        z = [ 2 * alpha * f_s,  2 * (1 - alpha) * f_f ]
    
    The scaling factor of 2.0 ensures that at equilibrium (alpha = 0.5),
    the expected feature magnitude is strictly conserved.
    """
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, f_s, f_f):
        """
        Args:
            f_s: Spatial feature vector (batch, 2048)
            f_f: Frequency feature vector (batch, 2048)
        Returns:
            fused_z: Gated concatenated representation (batch, 4096)
            alpha: Condition-aware dynamic routing weight (batch, 1)
        """
        combined = torch.cat([f_s, f_f], dim=1)
        alpha = self.gate(combined)
        gated_fs = 2.0 * alpha * f_s
        gated_ff = 2.0 * (1.0 - alpha) * f_f
        fused_z = torch.cat([gated_fs, gated_ff], dim=1)
        return fused_z, alpha


class GatedHybridDetector(nn.Module):
    """Complete Deepfake Detector equipped with Dynamic Gated Fusion."""
    def __init__(self, feature_dim=2048, dropout_rate=0.5, pretrained=True):
        super().__init__()
        self.spatial_branch = SpatialBranch(pretrained=pretrained)
        self.frequency_branch = FrequencyBranch(pretrained=pretrained)
        self.gated_fusion = DynamicGatedFusion(feature_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial_input, frequency_input, return_alpha=True):
        """
        Args:
            spatial_input: RGB image tensor (batch, 3, H, W)
            frequency_input: DCT coefficient tensor (batch, 3, H, W)
            return_alpha: Whether to return the dynamic routing weight
        Returns:
            prob: Probability of synthetic face (batch, 1)
            alpha: (optional) Dynamic gating weight (batch, 1)
        """
        f_s = self.spatial_branch(spatial_input)
        f_f = self.frequency_branch(frequency_input)
        fused, alpha = self.gated_fusion(f_s, f_f)
        prob = self.classifier(fused)

        if return_alpha:
            return prob, alpha
        return prob
