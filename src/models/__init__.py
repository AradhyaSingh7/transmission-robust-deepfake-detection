"""
Model Architectures for Dual-Stream Spatial-Frequency Deepfake Detection
"""
from .branches import SpatialBranch, FrequencyBranch
from .cbam import CBAM, ChannelAttention
from .detector import HybridDetector
from .dynamic_gate import DynamicGatedFusion, GatedHybridDetector

__all__ = [
    "SpatialBranch",
    "FrequencyBranch",
    "CBAM",
    "ChannelAttention",
    "HybridDetector",
    "DynamicGatedFusion",
    "GatedHybridDetector",
]
