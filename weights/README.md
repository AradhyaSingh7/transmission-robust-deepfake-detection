# Pretrained Model Checkpoints

This directory contains instructions for obtaining and organizing model weights.

### Weight Checkpoint Details
- **Baseline Architecture**: Dual-stream spatial-frequency network with CBAM attention.
- **Dynamic Gated Model**: Dual-stream network with input-conditioned adaptive gating.
- **Parameter Size**: ~196 MB per checkpoint (ResNet-50 backbones + projection heads).

### Instructions
1. Download `gated_model_best.pt` from the project repository's Release assets or shared drive.
2. Place the `.pt` file directly into this `weights/` folder:
   ```bash
   weights/gated_model_best.pt
   ```
3. If no checkpoint is present, `demo.py` automatically initializes architectural weights in demonstration mode for pipeline verification.
