"""
Convolutional Block Attention Module (CBAM) - Feature Dimension Attention
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention mechanism for fused multi-stream feature vectors.
    
    Dynamically weighs feature dimensions based on their relevance for
    synthetic artifact discrimination, adapting from Woo et al. (ECCV 2018).
    """
    def __init__(self, in_features, reduction_ratio=16):
        super().__init__()
        mid_features = max(in_features // reduction_ratio, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(inplace=True),
            nn.Linear(mid_features, in_features)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_features)
        Returns:
            Tensor of shape (batch, in_features) — channel-weighted features
        """
        weights = self.sigmoid(self.mlp(x))
        return x * weights


class CBAM(nn.Module):
    """CBAM wrapper for 1D fused multi-stream latent vectors."""
    def __init__(self, in_features, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_features, reduction_ratio)

    def forward(self, x):
        return self.channel_attention(x)
