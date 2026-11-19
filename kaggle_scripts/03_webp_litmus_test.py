#!/usr/bin/env python
# ============================================================================
# PHASE 2: WEBP LITMUS TEST + CALIBRATED ACCURACY ANALYSIS
# ============================================================================
# NO TRAINING. Uses existing checkpoints for:
#   1. Baseline (Full S+F+CBAM)
#   2. Naive Augmentation (Full S+F+CBAM trained with JPEG aug)
#   3. Cosine Consistency (Full S+F+CBAM, Spatial branch, lambda=0.1)
#
# Evaluates all 3 on:
#   - In-Distribution: JPEG Q=100, 75, 50, 25
#   - Out-of-Distribution (Litmus Test): WebP Q=75, 50, 25
#   - Calibration Analysis: Resize 75%, 50%
#   - Compound: Combined (Q50 + Res75%)
#
# Primary Metric: ROC-AUC (matching Kumar et al.'s base paper)
# Secondary: Default Accuracy (0.5), Calibrated Accuracy (optimal threshold)
#
# Estimated Runtime: ~15 minutes on T4 GPU
# ============================================================================

# %% Cell 1: Setup
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy", "pillow", "seaborn"])

# %% Cell 2: Imports & Configuration
import os, io, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────
# DATA ROOT (update if your Kaggle dataset name differs)
# ──────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/input/real-and-fake-face-detection/real_vs_fake/real-vs-fake")

# ──────────────────────────────────────────────────────────────────────────
# CHECKPOINT PATHS — UPDATE THESE TO MATCH YOUR KAGGLE INPUT DATASETS
# ──────────────────────────────────────────────────────────────────────────
# These should point to the model_best.pt files from your previous runs.
# Upload previous notebook outputs as Kaggle datasets and reference them here.
CKPT_PATHS = {
    # From your ablation v3 run (trained fresh with identical recipe)
    "Baseline": Path("/kaggle/input/ablation-v3-checkpoints/checkpoints/full/model_best.pt"),

    # From your consistency_training_cosine.py run
    "Naive Aug (JPEG)": Path("/kaggle/input/consistency-cosine-checkpoints/checkpoints/naive_aug/model_best.pt"),

    # From your consistency_training_cosine.py run (best: spatial lambda=0.1)
    "Consistency (Cosine)": Path("/kaggle/input/consistency-cosine-checkpoints/checkpoints/cosine_spatial_l01/model_best.pt"),
}

# ──────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/kaggle/working/phase2_litmus_test")
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
BENCH_PER_CLASS = 2000    # 2000 real + 2000 fake = 4000 images per condition
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

GLOBAL_START = time.time()
def elapsed():
    return f"[{(time.time() - GLOBAL_START) / 60:.1f}min elapsed]"

print(f"Device: {DEVICE}")
print(f"Data root exists: {DATA_ROOT.exists()}")
for name, p in CKPT_PATHS.items():
    status = "FOUND" if p.exists() else "MISSING"
    print(f"  {name:25s} -> {status} ({p})")

# %% Cell 3: Architecture (identical to your training scripts)
class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.resnet50(weights=None).children())[:-1])
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.resnet50(weights=None).children())[:-1])
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

class ChannelAttention(nn.Module):
    def __init__(self, in_features, reduction_ratio=16):
        super().__init__()
        mid = max(in_features // reduction_ratio, 64)
        self.mlp = nn.Sequential(nn.Linear(in_features, mid), nn.ReLU(True), nn.Linear(mid, in_features))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.mlp(x))

class CBAM(nn.Module):
    def __init__(self, in_features, reduction_ratio=16):
        super().__init__()
        self.ca = ChannelAttention(in_features, reduction_ratio)
    def forward(self, x):
        return self.ca(x)

class HybridDetector(nn.Module):
    def __init__(self, feature_dim=2048, dropout_rate=0.5):
        super().__init__()
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        self.cbam = CBAM(in_features=feature_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
    def forward(self, spatial_input, frequency_input):
        sf = self.spatial_branch(spatial_input)
        ff = self.frequency_branch(frequency_input)
        feats = self.cbam(torch.cat([sf, ff], dim=1))
        return self.classifier(feats)

# %% Cell 4: Transforms & Degradation Functions
SPATIAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class DCTTransform:
    def __call__(self, image):
        arr = np.array(image, dtype=np.float32)
        chans = []
        for c in range(3):
            dc = dctn(arr[:, :, c], type=2, norm="ortho")
            dl = np.sign(dc) * np.log1p(np.abs(dc))
            chans.append((dl - dl.min()) / (dl.max() - dl.min() + 1e-8))
        return torch.from_numpy(np.stack(chans, axis=0)).float()

DCT_TF = DCTTransform()

def jpeg_compress(image, quality):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def webp_compress(image, quality):
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

class ResizeDegradation:
    def __init__(self, scale, target_size=256):
        self.scale, self.target_size = scale, target_size
    def __call__(self, image):
        w, h = image.size
        small = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        return image.resize(small, Image.BILINEAR).resize((self.target_size, self.target_size), Image.BILINEAR)

class ComposeTransforms:
    def __init__(self, fns):
        self.fns = fns
    def __call__(self, img):
        for f in self.fns:
            img = f(img)
        return img

# %% Cell 5: Datasets
class BaseDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.image_size = image_size
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = Path(root_dir) / label_name
            files = sorted(f for f in class_dir.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            for p in files:
                self.samples.append((p, label_idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        return SPATIAL_TF(img), DCT_TF(img), torch.tensor(label, dtype=torch.float32)

class BenchDataset(Dataset):
    def __init__(self, samples, degrade_fn, image_size=256):
        self.samples = samples
        self.degrade_fn = degrade_fn
        self.image_size = image_size
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        if self.degrade_fn is not None:
            img = self.degrade_fn(img)
        return SPATIAL_TF(img), DCT_TF(img), torch.tensor(label, dtype=torch.float32)

def balanced_subset(dataset, n_per_class, seed=42):
    rng = random.Random(seed)
    real = [s for s in dataset.samples if s[1] == 0]
    fake = [s for s in dataset.samples if s[1] == 1]
    n = min(n_per_class, len(real), len(fake))
    subset = rng.sample(real, n) + rng.sample(fake, n)
    labels = np.array([s[1] for s in subset])
    assert (labels == 0).sum() == n and (labels == 1).sum() == n, "Unbalanced!"
    return subset, labels

# %% Cell 6: Evaluation Engine
def find_calibrated_accuracy(labels, probs):
    """Finds the maximum achievable accuracy by sweeping thresholds."""
    best_acc, best_t = 0.0, 0.5
    for t in np.linspace(0.001, 0.999, 999):
        acc = accuracy_score(labels, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_acc * 100.0, best_t

def compute_full_metrics(labels, probs):
    """Computes all metrics for one (model, condition) pair."""
    default_pred = (probs > 0.5).astype(int)
    cal_acc, cal_thresh = find_calibrated_accuracy(labels, probs)
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "default_acc": float(accuracy_score(labels, default_pred) * 100),
        "calibrated_acc": cal_acc,
        "calibrated_threshold": float(cal_thresh),
        "f1": float(f1_score(labels, default_pred, zero_division=0)),
        "mean_p_real": float(probs[real_mask].mean()),
        "mean_p_fake": float(probs[fake_mask].mean()),
    }

# %% Cell 7: Build Evaluation Conditions
def build_conditions():
    conds = {"Clean": None}
    # In-Distribution: JPEG (models saw this during training)
    for q in [100, 75, 50, 25]:
        conds[f"JPEG Q={q}"] = lambda img, q=q: jpeg_compress(img, q)
    # Out-of-Distribution: WebP (THE LITMUS TEST — no model ever saw this)
    for q in [75, 50, 25]:
        conds[f"WebP Q={q}"] = lambda img, q=q: webp_compress(img, q)
    # Calibration Analysis: Resize
    for s in [0.75, 0.5]:
        conds[f"Resize {int(s*100)}%"] = ResizeDegradation(s, IMAGE_SIZE)
    # Compound: Realistic social media pipeline
    conds["Combined (Q50+Res75%)"] = ComposeTransforms([
        lambda img: jpeg_compress(img, 50),
        ResizeDegradation(0.75, IMAGE_SIZE)
    ])
    return conds

CONDITIONS = build_conditions()

# %% Cell 8: Build Balanced Benchmark Set
print(f"\n[*] Loading test dataset... {elapsed()}")
test_ds = BaseDataset(DATA_ROOT / "test", image_size=IMAGE_SIZE)
bench_samples, bench_labels = balanced_subset(test_ds, BENCH_PER_CLASS, seed=SEED)
print(f"    Benchmark set: {len(bench_labels)} images "
      f"({int((bench_labels == 0).sum())} real / {int((bench_labels == 1).sum())} fake)")

# %% Cell 9: Load Models & Evaluate
@torch.no_grad()
def predict_all(model, samples, degrade_fn):
    dl = DataLoader(BenchDataset(samples, degrade_fn, IMAGE_SIZE),
                    batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_probs = []
    for s, f, _ in dl:
        p = model(s.to(DEVICE), f.to(DEVICE)).squeeze(1).cpu().numpy()
        all_probs.append(p)
    return np.concatenate(all_probs)

all_results = {}

for model_name, ckpt_path in CKPT_PATHS.items():
    if not ckpt_path.exists():
        print(f"\n[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
        continue

    print(f"\n{'='*70}")
    print(f"  EVALUATING: {model_name}  {elapsed()}")
    print(f"{'='*70}")

    model = HybridDetector()
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    # Handle older checkpoint naming convention
    sd = {k.replace("cbam.channel_attention.", "cbam.ca."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    all_results[model_name] = {}
    print(f"  {'Condition':<24s} | {'AUC':>7s} | {'Def.Acc':>8s} | {'Cal.Acc':>8s} | {'Cal.Thr':>7s} | {'P(Real)':>8s} | {'P(Fake)':>8s}")
    print(f"  {'-'*90}")

    for cond_name, fn in CONDITIONS.items():
        probs = predict_all(model, bench_samples, fn)
        metrics = compute_full_metrics(bench_labels, probs)
        all_results[model_name][cond_name] = metrics
        print(f"  {cond_name:<24s} | {metrics['auc']:7.4f} | {metrics['default_acc']:7.1f}% | "
              f"{metrics['calibrated_acc']:7.1f}% | {metrics['calibrated_threshold']:7.3f} | "
              f"{metrics['mean_p_real']:8.4f} | {metrics['mean_p_fake']:8.4f}")

    del model
    torch.cuda.empty_cache()

# Save all results
with open(RESULTS_DIR / "phase2_litmus_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'phase2_litmus_results.json'}")

# %% Cell 10: Summary Tables
models = list(all_results.keys())
conds = list(CONDITIONS.keys())

# ─── TABLE 1: ROC-AUC (PRIMARY METRIC) ───
print(f"\n{'='*100}")
print("  TABLE 1: ROC-AUC (PRIMARY METRIC — matches Kumar et al.)")
print(f"{'='*100}")
hdr = f"{'Condition':<24s}" + "".join(f" | {m:>22s}" for m in models)
print(hdr); print("-" * len(hdr))
for c in conds:
    row = f"{c:<24s}"
    vals = [all_results[m][c]["auc"] for m in models]
    best = max(vals)
    for m, v in zip(models, vals):
        marker = " *" if v == best else "  "
        row += f" | {v:20.4f}{marker}"
    print(row)
print("  (* = best per condition)")

# ─── TABLE 2: THE LITMUS TEST COMPARISON ───
print(f"\n{'='*100}")
print("  TABLE 2: THE LITMUS TEST — JPEG (In-Distribution) vs WebP (Out-of-Distribution)")
print(f"{'='*100}")
print(f"{'Model':<25s} | {'JPEG Q=75':>10s} {'Q=50':>8s} {'Q=25':>8s} | {'WebP Q=75':>10s} {'Q=50':>8s} {'Q=25':>8s} | {'Avg Drop':>9s}")
print("-" * 100)
for m in models:
    j = [all_results[m][f"JPEG Q={q}"]["auc"] for q in [75, 50, 25]]
    w = [all_results[m][f"WebP Q={q}"]["auc"] for q in [75, 50, 25]]
    avg_drop = np.mean([w[i] - j[i] for i in range(3)])
    print(f"{m:<25s} | {j[0]:10.4f} {j[1]:8.4f} {j[2]:8.4f} | {w[0]:10.4f} {w[1]:8.4f} {w[2]:8.4f} | {avg_drop:+8.4f}")

# ─── TABLE 3: CALIBRATION ANALYSIS (Proving Resize is just a threshold shift) ───
print(f"\n{'='*100}")
print("  TABLE 3: RESIZE CALIBRATION PROOF (Accuracy @ 0.5 vs Optimal Threshold)")
print(f"{'='*100}")
resize_conds = ["Resize 75%", "Resize 50%"]
print(f"{'Model':<25s} | {'Condition':<14s} | {'AUC':>7s} | {'Acc@0.5':>8s} | {'Acc@Best':>9s} | {'Best Thr':>9s} | {'Recovery':>9s}")
print("-" * 100)
for m in models:
    for rc in resize_conds:
        r = all_results[m][rc]
        recovery = r["calibrated_acc"] - r["default_acc"]
        print(f"{m:<25s} | {rc:<14s} | {r['auc']:7.4f} | {r['default_acc']:7.1f}% | "
              f"{r['calibrated_acc']:8.1f}% | {r['calibrated_threshold']:9.3f} | {recovery:+8.1f} pp")

# %% Cell 11: Publication-Ready Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = {"Baseline": "#E53935", "Naive Aug (JPEG)": "#FF9800", "Consistency (Cosine)": "#2196F3"}
markers_j = {"Baseline": "o", "Naive Aug (JPEG)": "o", "Consistency (Cosine)": "o"}
markers_w = {"Baseline": "s", "Naive Aug (JPEG)": "s", "Consistency (Cosine)": "s"}
qs = [75, 50, 25]

# ─── Plot 1: JPEG AUC (In-Distribution) ───
ax = axes[0]
for m in models:
    vals = [all_results[m][f"JPEG Q={q}"]["auc"] for q in qs]
    ax.plot(qs, vals, "o-", label=m, color=colors.get(m, "gray"), lw=2.5, markersize=8)
ax.set_title("In-Distribution: JPEG", fontsize=12, fontweight="bold")
ax.set_xlabel("Quality Factor Q"); ax.set_ylabel("ROC-AUC")
ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
ax.set_ylim(0.65, 1.01)

# ─── Plot 2: WebP AUC (Out-of-Distribution — THE LITMUS TEST) ───
ax = axes[1]
for m in models:
    vals = [all_results[m][f"WebP Q={q}"]["auc"] for q in qs]
    ax.plot(qs, vals, "s--", label=m, color=colors.get(m, "gray"), lw=2.5, markersize=8)
ax.set_title("Out-of-Distribution: WebP (LITMUS TEST)", fontsize=12, fontweight="bold")
ax.set_xlabel("Quality Factor Q"); ax.set_ylabel("ROC-AUC")
ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
ax.set_ylim(0.65, 1.01)

# ─── Plot 3: AUC Drop from JPEG to WebP (The Smoking Gun) ───
ax = axes[2]
x = np.arange(3)
width = 0.25
for i, m in enumerate(models):
    j_auc = [all_results[m][f"JPEG Q={q}"]["auc"] for q in qs]
    w_auc = [all_results[m][f"WebP Q={q}"]["auc"] for q in qs]
    drops = [w_auc[k] - j_auc[k] for k in range(3)]
    ax.bar(x + i * width, drops, width, label=m, color=colors.get(m, "gray"), edgecolor="k", lw=0.5)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x + width); ax.set_xticklabels([f"Q={q}" for q in qs])
ax.set_ylabel("AUC Change (WebP - JPEG)")
ax.set_title("AUC Drop: JPEG -> WebP (Generalization Gap)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

plt.suptitle("Phase 2: Does Naive Aug Memorize JPEG, or Does Consistency Generalize?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "litmus_test_jpeg_vs_webp.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── Plot 4: Calibration Recovery on Resize ───
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
width = 0.25

for i, rc in enumerate(["Resize 75%", "Resize 50%"]):
    def_accs = [all_results[m][rc]["default_acc"] for m in models]
    cal_accs = [all_results[m][rc]["calibrated_acc"] for m in models]
    offset = (i - 0.5) * width * 2.5

    bars_def = ax.bar(x + offset - width/2, def_accs, width, label=f"{rc} (Acc@0.5)" if i == 0 else "",
                      color="#E53935" if i == 0 else "#FF7043", alpha=0.8, edgecolor="k", lw=0.5)
    bars_cal = ax.bar(x + offset + width/2, cal_accs, width, label=f"{rc} (Calibrated)" if i == 0 else "",
                      color="#4CAF50" if i == 0 else "#66BB6A", alpha=0.8, edgecolor="k", lw=0.5)

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Resize: Default (0.5) vs Calibrated Threshold Accuracy\n"
             "Proves resize is a threshold shift, not representation collapse", fontsize=12, fontweight="bold")
ax.legend(["Acc @ 0.5 threshold", "Acc @ Optimal threshold"], fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "resize_calibration_proof.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Cell 12: Key Findings Summary
print(f"\n{'='*70}")
print(f"  KEY FINDINGS  {elapsed()}")
print(f"{'='*70}")

print("\n1. LITMUS TEST RESULTS (JPEG vs WebP ROC-AUC):")
for m in models:
    j_avg = np.mean([all_results[m][f"JPEG Q={q}"]["auc"] for q in [75, 50, 25]])
    w_avg = np.mean([all_results[m][f"WebP Q={q}"]["auc"] for q in [75, 50, 25]])
    gap = w_avg - j_avg
    print(f"   {m:25s}: JPEG avg AUC = {j_avg:.4f} | WebP avg AUC = {w_avg:.4f} | Gap = {gap:+.4f}")

if len(models) >= 3:
    naive_gap = np.mean([all_results["Naive Aug (JPEG)"][f"WebP Q={q}"]["auc"] -
                         all_results["Naive Aug (JPEG)"][f"JPEG Q={q}"]["auc"] for q in [75, 50, 25]])
    consis_gap = np.mean([all_results["Consistency (Cosine)"][f"WebP Q={q}"]["auc"] -
                          all_results["Consistency (Cosine)"][f"JPEG Q={q}"]["auc"] for q in [75, 50, 25]])
    print(f"\n   Naive Aug generalization gap:    {naive_gap:+.4f}")
    print(f"   Consistency generalization gap:  {consis_gap:+.4f}")
    if consis_gap > naive_gap:
        print("   -> Consistency generalizes BETTER to unseen codecs (smaller drop on WebP)")
    else:
        print("   -> Naive Aug generalizes comparably or better (investigate further)")

print("\n2. RESIZE CALIBRATION PROOF:")
for m in models:
    r50 = all_results[m]["Resize 50%"]
    print(f"   {m:25s}: AUC = {r50['auc']:.4f} | Acc@0.5 = {r50['default_acc']:.1f}% | "
          f"Acc@Best = {r50['calibrated_acc']:.1f}% (threshold = {r50['calibrated_threshold']:.3f})")

print(f"\n{'='*70}")
print(f"  PHASE 2 COMPLETE  {elapsed()}")
print(f"{'='*70}")

# ============================================================================
# NOTES
# ============================================================================
# 1. The primary metric is ROC-AUC, matching Kumar et al.'s base paper.
# 2. WebP uses different compression (intra-prediction + entropy coding) than
#    JPEG (8x8 block DCT quantization). If Naive Aug memorized JPEG block
#    patterns, its AUC will drop on WebP. If Consistency learned true
#    representation invariance, it will hold up.
# 3. The calibrated accuracy for resize proves the model's discriminative
#    power is intact — only the probability calibration shifts.
# 4. No training was performed in this script. All models used existing
#    checkpoints from previous experiments.
# ============================================================================
