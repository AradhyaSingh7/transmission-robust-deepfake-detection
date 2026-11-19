"""
Spatial and Frequency Feature Extraction Branches
"""
import torch
import torch.nn as nn
from torchvision import models


class SpatialBranch(nn.Module):
    """Spatial feature extraction branch.
    
    Processes RGB pixel arrays through a ResNet-50 backbone to capture
    semantic, textural, boundary, and blend artifacts.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Remove original 1000-class FC layer; keep global average pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 3, H, W) — normalized RGB image
        Returns:
            Tensor of shape (batch, 2048) — spatial representation vector
        """
        feats = self.features(x)  # (batch, 2048, 1, 1)
        return feats.squeeze(-1).squeeze(-1)  # (batch, 2048)


class FrequencyBranch(nn.Module):
    """Frequency feature extraction branch.
    
    Processes 2D Discrete Cosine Transform (DCT) log-magnitude spectral
    coefficients through a ResNet-50 backbone to detect high-frequency GAN
    upsampling anomalies and grid discrepancies.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Remove original 1000-class FC layer; keep global average pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 3, H, W) — log-scaled DCT coefficients
        Returns:
            Tensor of shape (batch, 2048) — spectral representation vector
        """
        feats = self.features(x)  # (batch, 2048, 1, 1)
        return feats.squeeze(-1).squeeze(-1)  # (batch, 2048)
