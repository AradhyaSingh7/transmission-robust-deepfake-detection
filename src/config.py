"""
Configuration and Hyperparameters
"""
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
WEIGHTS_DIR = BASE_DIR / "weights"
RESULTS_DIR = BASE_DIR / "results"

# Image processing parameters
IMAGE_SIZE = 256
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Model parameters
FEATURE_DIM = 2048
DROPOUT_RATE = 0.5
DEFAULT_THRESHOLD = 0.5

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Degradation conditions audited in paper
DEGRADATION_PRESETS = {
    "clean": {"type": "none", "description": "Unperturbed operating baseline"},
    "jpeg_q100": {"type": "jpeg", "quality": 100, "description": "Lossless JPEG encoding"},
    "jpeg_q75": {"type": "jpeg", "quality": 75, "description": "Standard social media compression (proxy for FF++ C23)"},
    "jpeg_q50": {"type": "jpeg", "quality": 50, "description": "Aggressive lossy JPEG compression"},
    "jpeg_q25": {"type": "jpeg", "quality": 25, "description": "Severe lossy JPEG compression"},
    "webp_q75": {"type": "webp", "quality": 75, "description": "Modern WebP codec transcoding"},
    "webp_q25": {"type": "webp", "quality": 25, "description": "Out-of-domain WebP Litmus test (shortcut detector)"},
    "resize_75": {"type": "resize", "scale": 0.75, "description": "Geometric downsampling to 75% resolution"},
    "resize_50": {"type": "resize", "scale": 0.50, "description": "Severe geometric downsampling to 50% resolution"},
    "compound": {"type": "compound", "jpeg_quality": 50, "resize_scale": 0.75, "description": "Compound stress: JPEG Q=50 + Resize 75%"},
}
