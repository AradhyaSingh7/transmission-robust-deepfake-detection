# Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Evaluation CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

> Official PyTorch implementation and benchmark suite for the research paper:  
> **"Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion"**  
> *Aradhya Singh & Khushi Sagar — Netaji Subhas University of Technology (NSUT)*

---

## 📌 Overview

Dual-stream spatial–frequency architectures effectively identify synthetic facial images in ideal, uncompressed environments. However, real-world deployment on communication channels (e.g., social media uploads) introduces lossy compression and resizing that cause severe performance degradation.

This repository provides an audited experimental audit and the **Dynamic Gated Fusion** framework, which dynamically re-weights spatial and frequency streams based on input condition:

```
Input Image (x)
  ├── Spatial Stream:   x        ──► ResNet-50 ──► f_s (2048-d) ──┐
  └── Frequency Stream: DCT(x)   ──► ResNet-50 ──► f_f (2048-d) ──┤
                                                                   ▼
                                                       Dynamic Gated Routing
                                                       α = Sigmoid(MLP([f_s, f_f]))
                                                       z = [2α·f_s, 2(1-α)·f_f]
                                                                   ▼
                                                          MLP Classifier Head
                                                                   ▼
                                                            Prediction: Real / Fake
```

---

## 🔬 Key Empirical Discoveries

1. **Asymmetric Branch Dissociation:** Under lossy JPEG compression, spatial models collapse into extreme fake-bias (True Negative Rate crashes to 1.65% at Q=50), whereas geometric resizing abolishes frequency spectral patterns (True Positive Rate drops to 0.15% at 50% downsampling).
2. **The WebP Litmus Test (Shortcut Detection):** Naive JPEG data augmentation learns a codec-specific artifact shortcut. When evaluated on unseen WebP compression (Q=25), naive augmentation crashes to 68.53% accuracy, whereas our consistency and dynamic gated models preserve representation invariance.
3. **Dynamic Gated Fusion Dominance:** The adaptive gating mechanism delivers massive classification accuracy gains (+40.45% at JPEG Q=75, +42.43% at JPEG Q=50, and +28.25% under 50% resizing) with negligible computational overhead (+0.05% FLOPs).

---

## 📊 Audited Benchmark Results

Evaluation conducted on an audited benchmark of $N = 4{,}000$ facial images (2,000 FFHQ Real vs. 2,000 StyleGAN Synthetic):

| Operational Regime | Baseline Acc | Dynamic Gated Acc | Delta (Δ) | Baseline AUC | Gated AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 99.05% | 99.45% | **+0.40%** | 0.9994 | 1.0000 |
| **JPEG Q=100** | 98.60% | 99.12% | **+0.53%** | 0.9992 | 1.0000 |
| **JPEG Q=75 (Social Media)** | 57.08% | 97.52% | **+40.45%** | 0.9493 | 0.9987 |
| **JPEG Q=50 (Aggressive)** | 51.70% | 94.12% | **+42.42%** | 0.9191 | 0.9899 |
| **JPEG Q=25 (Severe)** | 50.83% | 82.67% | **+31.85%** | 0.8742 | 0.9333 |
| **WebP Q=75 (Web Codec)** | 86.38% | 95.28% | **+8.90%** | 0.9525 | 0.9963 |
| **WebP Q=25 (Out-of-Domain)** | 77.62% | 77.92% | **+0.30%** | 0.8570 | 0.9133 |
| **Resize 75% (Downsample)** | 78.65% | 95.45% | **+16.80%** | 0.9800 | 0.9972 |
| **Resize 50% (Severe)** | 53.88% | 82.12% | **+28.25%** | 0.9221 | 0.9678 |
| **Compound (Q50 + Res75%)** | 67.65% | 93.08% | **+25.42%** | 0.9443 | 0.9836 |

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/AradhyaSingh7/transmission-robust-deepfake-detection.git
cd transmission-robust-deepfake-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Single-Image Inference CLI

Test the executable detector on a sample facial image:

```bash
# Clean inference
python demo.py --image samples/fake/fake_01.jpg

# Simulate lossy social media compression (JPEG Q=50)
python demo.py --image samples/fake/fake_01.jpg --degrade jpeg50

# Run sequential audit across all sample images
python demo.py --sample-all
```

The script outputs a diagnostic console log and generates `demo_output.png` containing the RGB image, 2D DCT log-spectrum, and dynamic routing allocation.

### 3. Reproduce Paper Benchmarks

To reproduce all 10 operational benchmarks and export `BENCHMARK_SUMMARY.md`:

```bash
python evaluate.py
```

### 4. Run Unit Tests

```bash
python -m unittest discover -s tests
```

---

## 📂 Repository Structure

```text
transmission-robust-deepfake-detection/
├── paper/
│   └── Transmission_Robust_Deepfake_Detection_Paper.pdf  # Camera-ready paper PDF
├── docs/
│   └── figures/                    # High-resolution publication figures
├── samples/
│   ├── real/                       # Sample FFHQ real face images
│   └── fake/                       # Sample StyleGAN synthetic face images
├── weights/
│   └── README.md                   # Instructions for model checkpoints
├── results/
│   ├── baseline_ablation_results.json
│   ├── dynamic_gated_fusion_results.json
│   └── litmus_result.txt
├── src/
│   ├── models/
│   │   ├── branches.py             # Spatial & Frequency ResNet-50 backbones
│   │   ├── cbam.py                 # CBAM channel attention module
│   │   ├── detector.py             # Static baseline detector
│   │   └── dynamic_gate.py         # Dynamic Gated Fusion implementation
│   ├── data/
│   │   ├── transforms.py           # 2D DCT log-spectrum & lossy degradations
│   │   └── dataset.py              # Paired dataset loader
│   ├── evaluation/
│   │   └── benchmark.py            # Comparative evaluation loader
│   └── utils/
│       └── metrics.py              # Accuracy, ROC-AUC, TNR/TPR calculation
├── demo.py                         # Standalone executable CLI inference script
├── evaluate.py                     # Master benchmark reproduction script
├── requirements.txt                # Dependency specifications
├── setup.py                        # Pip package setup
└── README.md                       # Documentation
```

---

## 📜 Citation

If you find this work or implementation helpful in your research, please cite:

```bibtex
@article{singh2026transmission,
  title     = {Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion},
  author    = {Singh, Aradhya and Sagar, Khushi},
  journal   = {Department of Computer Science and Engineering, Netaji Subhas University of Technology},
  year      = {2026}
}
```

---

## 📄 License
This project is licensed under the MIT License.
