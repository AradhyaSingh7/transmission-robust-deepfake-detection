# Cloud Training & Kaggle Experiment Scripts

This directory contains the self-contained experiment pipelines executed on Kaggle GPU accelerators (NVIDIA Tesla T4 / P100) across the research progression of the paper.

---

## 📋 Experiment Progression & Script Manifest

| Script | Research Phase | Objective | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **`01_baseline_ablation_study.py`** | Phase 1: Baseline Architecture | 4-way architectural ablation (`full`, `spatial_only`, `frequency_only`, `fusion_no_attention`) under JPEG compression | `ablation_v3_results_full.json`, model checkpoints |
| **`02_cosine_consistency_training.py`** | Phase 2: Invariance Defense | Trains dual-stream backbone with Latent Cosine Consistency Regularization ($\lambda = 0.1$) | `consistency_cosine_best.pt`, loss curves |
| **`03_webp_litmus_test.py`** | Phase 2: Shortcut Learning Audit | Evaluates Clean Baseline vs. Naive JPEG Augmentation vs. Cosine Consistency on unseen WebP codec ($Q=25$) | `litmus_result.txt` (score separation analysis) |
| **`04_dynamic_gated_fusion.py`** | Phase 3: Architectural Breakthrough | Trains Dynamic Gating Module ($z = [2\alpha f_s, 2(1-\alpha)f_f]$) on mixed stress & evaluates across all 10 operational conditions | `gated_model_best.pt`, `dynamic_gated_fusion_results.json` |

---

## ⚙️ Kaggle Execution Protocol

1. **Dataset Mounting:**  
   Mount the standard 140k Real and Fake Faces benchmark:
   ```text
   Kaggle Dataset: xhlulu/140k-real-and-fake-faces
   Path: /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake
   ```

2. **Environment & Hardware:**  
   - Accelerator: GPU (Tesla T4 x2 or P100)
   - Persistence: Enable internet for initial torchvision weights download
   - PyTorch: >= 2.0 with CUDA 11.8+

3. **Running in Kaggle Notebook:**  
   Each script can be pasted directly into a Kaggle Code Cell or uploaded as a `.py` script and run:
   ```bash
   python 04_dynamic_gated_fusion.py
   ```
