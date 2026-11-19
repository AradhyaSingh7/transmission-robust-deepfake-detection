# Audited Paper Benchmark Summary

**Paper:** Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion  
**Authors:** Aradhya Singh & Khushi Sagar (Netaji Subhas University of Technology)  

### Table 1: Dual-Stream Baseline vs. Dynamic Gated Fusion

| Operational Condition | Baseline Acc | Gated Acc | Delta Acc | Baseline AUC | Gated AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean (Operating Base) | 99.05% | 99.45% | **+0.40%** | 0.9994 | 1.0000 |
| JPEG Q=100 (Lossless) | 98.60% | 99.12% | **+0.53%** | 0.9992 | 1.0000 |
| JPEG Q=75 (Social Media) | 57.08% | 97.52% | **+40.45%** | 0.9493 | 0.9987 |
| JPEG Q=50 (Aggressive) | 51.70% | 94.12% | **+42.42%** | 0.9191 | 0.9899 |
| JPEG Q=25 (Severe) | 50.83% | 82.67% | **+31.85%** | 0.8742 | 0.9333 |
| WebP Q=75 (Web Codec) | 86.38% | 95.28% | **+8.90%** | 0.9525 | 0.9963 |
| WebP Q=25 (Out-of-Domain) | 77.62% | 77.92% | **+0.30%** | 0.8570 | 0.9133 |
| Resize 75% (Downsample) | 78.65% | 95.45% | **+16.80%** | 0.9800 | 0.9972 |
| Resize 50% (Severe) | 53.88% | 82.12% | **+28.25%** | 0.9221 | 0.9678 |
| Compound (Q50 + Res75%) | 67.65% | 93.08% | **+25.42%** | 0.9443 | 0.9836 |


*Evaluation conducted on audited class-balanced benchmark of N = 4,000 images (2,000 FFHQ Real, 2,000 StyleGAN Synthetic).*
