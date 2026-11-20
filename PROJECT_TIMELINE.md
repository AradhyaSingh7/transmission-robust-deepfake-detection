# Project Evolution & Research Progression Timeline

**Project Title:** Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion  
**Authors:** Aradhya Singh & Khushi Sagar  
**Affiliation:** Department of Computer Science and Engineering, Netaji Subhas University of Technology (NSUT) 
**Reference Foundation:** Kumar et al. (2026), *Scientific Reports* (Nature Portfolio), Vol. 16, Art. 16612  
**Evaluation Benchmark:** 140k Real and Fake Faces Benchmark (FFHQ Real vs. StyleGAN Synthetic)  

---

## 🗺️ Chronological Progression Arc

```
AUG 2026                  SEP 2026                 OCT 2026                  NOV 2026
┌───────────────────────┐ ┌──────────────────────┐ ┌───────────────────────┐ ┌────────────────────────┐
│ Phase 1: Replication  │ │ Phase 2: Stress-Test │ │ Phase 3 & 4: Litmus   │ │ Phase 5 & 6: Gated     │
│ & Baseline Validation │►│ & Failure Diagnosis  │►│ Test & Regularization │►│ Fusion, Paper & Packaging│
│ (99%+ Clean Ceiling)  │ │ (Branch Dissociation)│ │ (Shortcut Trap Proof) │ │ (+42.4% Win, IEEE PDF) │
└───────────────────────┘ └──────────────────────┘ └───────────────────────┘ └────────────────────────┘
```

---

## Phase 1: Theoretical Inception & Baseline Replication
**Timeline:** August 1 – August 22, 2026  
**Primary Objective:** Replicate and verify published multi-stream state-of-the-art results on clean facial benchmarks.

* **Aug 1 – Aug 8:** **Literature Survey & Base Architecture Selection**  
  Conducted a literature survey across facial forensics literature. Selected the dual-stream model by *Kumar et al. (Nature Scientific Reports, 2026)* as our reference foundation: an RGB Spatial ResNet-50 paired with a 2D-DCT Frequency ResNet-50 combined via static CBAM channel attention.
* **Aug 9 – Aug 15:** **Mathematical & Preprocessing Pipeline Formulation**  
  Implemented the 2D Discrete Cosine Transform (DCT) log-magnitude pipeline with dynamic-range compression ($\text{sign}(C) \cdot \log(1 + |C|)$) and channel-wise min-max normalization into $[0, 1]$.
* **Aug 16 – Aug 22:** **Baseline Training & Clean Ceiling Reproduction**  
  Trained all four canonical architecture variants (`spatial_only`, `frequency_only`, `fusion_no_attention`, and `full_cbam`) on the 140k benchmark. Successfully reproduced state-of-the-art numbers on unperturbed clean imagery: **99.05% classification accuracy and 0.9994 ROC-AUC**.

---

## Phase 2: Operational Stress-Testing & Branch Dissociation Discovery
**Timeline:** August 23 – September 20, 2026  
**Primary Objective:** Audit multi-stream detector resilience under real-world social media transmission channels.

* **Aug 23 – Sep 3:** **Transmission Degradation Suite Design**  
  Formulated a standardized 10-condition degradation suite modeling real-world lossy transcoders (WhatsApp, Instagram, Telegram, X/Twitter): lossless JPEG ($Q=100$), social media proxy ($Q=75$), aggressive compression ($Q=50$), low-bandwidth ($Q=25$), geometric downsampling ($75\%, 50\%$), compound stress, and out-of-domain codecs.
* **Sep 4 – Sep 12:** **Class-Balanced Evaluation Protocol ($N = 4{,}000$)**  
  Instituted an audited, fixed test benchmark of 2,000 FFHQ Real and 2,000 StyleGAN Synthetic faces (Seed 42) to eliminate class-imbalance variance and guarantee statistically rigorous metric evaluation.
* **Sep 13 – Sep 20:** **Discovery of Asymmetric Branch Dissociation**  
  Identified a structural vulnerability in dual-stream detectors:
  * **Lossy JPEG Compression:** Collapses the spatial branch into an extreme *Fake-bias* (calling real faces fake). At $Q=50$, spatial True Negative Rate (TNR) crashes from 99.40% down to **1.65%**, reducing default accuracy to random chance (50.83%).
  * **Geometric Resizing:** Inversely collapses the frequency branch into a *Real-bias* (calling fake faces real), crashing True Positive Rate (TPR) down to **0.15%** at 50% downsampling.
  * **Conclusion:** Proved that static fusion cannot withstand asymmetric channel noise.

---

## Phase 3: Latent Feature Regularization
**Timeline:** September 21 – October 12, 2026  
**Primary Objective:** Enforce representation invariance under lossy compression without destroying clean discriminability.

* **Sep 21 – Sep 30:** **Loss Formulation & Metric Exploration**  
  Formulated a representation-invariance regularization loss rather than superficial pixel augmentation:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \mathcal{D}_{\text{consistency}}\big(f(\mathbf{x}), f(\mathbf{x}_{\text{degraded}})\big)$$
  Compared Mean Squared Error (MSE) against Cosine Distance across penalty weights $\lambda \in \{0.01, 0.1, 1.0\}$.
* **Oct 1 – Oct 12:** **Empirical Invariance Convergence**  
  Established that **Latent Cosine Consistency at $\lambda = 0.1$** offered the superior Pareto frontier: recovering JPEG $Q=50$ accuracy to **83.95%** and stabilizing the feature manifold, whereas MSE suffered from norm-collapse.

---

## Phase 4: The WebP Litmus Test (Shortcut Learning Exposed)
**Timeline:** October 13 – October 26, 2026  
**Primary Objective:** Audit whether standard data augmentation truly learns invariance or exploits superficial shortcuts.

* **Oct 13 – Oct 18:** **Designing the Out-of-Domain Challenge**  
  Posed the critical question: *Does naive JPEG augmentation generalize, or does it merely memorize the 8×8 DCT grid?* Designed the **WebP Litmus Test** by evaluating models on an out-of-domain predictive codec (WebP $Q=75, 50, 25$) unseen during training.
* **Oct 19 – Oct 26:** **Empirical Proof of Shortcut Failure**  
  * **Naive JPEG Augmentation:** Suffered an out-of-domain collapse. On WebP $Q=25$, its score separation ($\Delta = \mathbb{E}[P_{\text{fake}}] - \mathbb{E}[P_{\text{real}}]$) crashed to **0.375**, falling *below* the unaugmented baseline (0.505), proving it had learned a brittle codec-specific artifact.
  * **Latent Cosine Consistency:** Maintained true representation invariance with **0.526 score separation** and **0.9332 ROC-AUC**.
  * **Outcome:** Naive data augmentation was formally proven to be an untrustworthy shortcut and discarded.

---

## Phase 5: Architectural Breakthrough — Dynamic Gated Fusion
**Timeline:** October 27 – November 10, 2026  
**Primary Objective:** Solve the multi-stream bottleneck where one corrupted branch ruins joint static fusion.

* **Oct 27 – Nov 2:** **Mathematical Formulation & Module Design**  
  Recognized that static 50/50 concatenation forces the classifier head to ingest noise when one branch degrades. Formulated an input-conditioned, adaptive gating mechanism:
  $$\alpha = \sigma\big(\text{MLP}([f_s, f_f])\big), \qquad \mathbf{z} = \big[2\alpha f_s, \; 2(1-\alpha)f_f\big]$$
  The scaling factor of $2.0$ ensures expected feature magnitude conservation at equilibrium ($\alpha = 0.5$).
* **Nov 3 – Nov 10:** **Mixed-Manifold Training & Benchmark Dominance**  
  Froze the pretrained backbones to eliminate catastrophic forgetting and trained the gate and classifier on a balanced mixture of clean, compressed, and resized samples:
  * **+40.45%** accuracy gain at JPEG $Q=75$ ($97.52\%$ vs. $57.08\%$)
  * **+42.43%** accuracy gain at JPEG $Q=50$ ($94.12\%$ vs. $51.70\%$)
  * **+28.25%** accuracy gain at Resize 50% ($82.12\%$ vs. $53.88\%$)
  * **+25.42%** accuracy gain on Compound Stress ($93.08\%$ vs. $67.65\%$)
  * Negligible computational overhead: **+0.05% FLOPs** over static fusion.

---

## Phase 6: Formal Verification & Production Packaging
**Timeline:** November 11 – November 19, 2026  
**Primary Objective:** Compile final IEEE paper and package a reproducible open-source software release.

* **Nov 11 – Nov 16:** **Paper Authoring & Compilation**  
  Authored and typeset the complete research paper: *"Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion"*. Compiled publication-quality figures, Grad-CAM saliency maps, and DCT spectra into the final PDF.
* **Nov 17 – Nov 19:** **Production Packaging & CI/CD Pipeline**  
  Packaged the standalone submission repository:
  * `demo.py`: Standalone CLI inference generating 3-panel visual forensic plots.
  * `evaluate.py`: Automated benchmark reproduction generating Tables 1, 2, and 3.
  * `train.py`: Local CPU/GPU training engine with smoke-testing support.
  * `kaggle_scripts/`: All 4 full-scale cloud GPU training pipelines.
  * `tests/test_pipeline.py`: Automated unit tests integrated with GitHub Actions CI.

---

## 📊 Summary of Project Milestones & Deliverables

| Milestone | Deliverable | Status |
| :--- | :--- | :---: |
| **M1: Baseline Replication** | Clean benchmark validation ($99.05\%$ accuracy, $0.9994$ AUC) | ✅ Verified |
| **M2: Degradation Audit** | Asymmetric branch dissociation diagnosis on $N = 4{,}000$ images | ✅ Verified |
| **M3: Consistency Defense** | Latent cosine regularization ($\lambda=0.1$) implementation | ✅ Verified |
| **M4: Shortcut Audit** | The WebP Litmus Test proving naive augmentation failure | ✅ Verified |
| **M5: Dynamic Architecture** | Dynamic Gated Fusion module delivering double-digit accuracy gains | ✅ Verified |
| **M6: Academic Deliverables** | IEEE Format paper PDF + production-grade GitHub repo with CI | ✅ Complete |
