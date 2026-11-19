#!/usr/bin/env python
# ============================================================================
# ABLATION STUDY v3 — FIXED BENCHMARK (balanced test set)
# ============================================================================
# FIX APPLIED: v2 benchmark evaluated degraded images on the first N samples
# of the test dataset, which were ALL real (dataset is ordered real-first).
# This version creates a separate balanced test dataset for benchmarking
# (max_samples=BENCHMARK_TEST_SAMPLES per class), matching the consistency
# training notebooks. Training code is unchanged from v2.
#
# Time budget: ~1 hour on T4 (benchmark only, all models skip training)
# ============================================================================

# %% [markdown]
# # 🔬 Ablation Study v3: Fixed Benchmark with Balanced Test Set
#
# | Variant | Architecture | What It Isolates |
# |---|---|---|
# | **S-only** | Spatial branch alone → classifier | Spatial robustness |
# | **F-only** | Frequency branch alone → classifier | Frequency vulnerability |
# | **S+F** | Both branches fused, no CBAM | Fusion benefit |
# | **S+F+CBAM** | Full model | Attention benefit |

# %% [markdown]
# ## Cell 1: Setup

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])

# %% [markdown]
# ## Cell 2: Imports & Config

# %%
import os, json, time, random, io, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake")

# Output
OUTPUT_DIR = Path("/kaggle/working/ablation_study")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Copy trained variant checkpoints from input dataset
import shutil
variants_input = Path("/kaggle/input/datasets/aradhyasingh16/ablation-study-train-output/ablation_study/checkpoints")
if variants_input.exists():
    shutil.copytree(str(variants_input), str(CHECKPOINT_DIR), dirs_exist_ok=True)
    print("✅ Restored variant checkpoints")
    for m in ["spatial_only", "frequency_only", "fusion_no_attention"]:
        status = "✅" if (CHECKPOINT_DIR / m / "model_best.pt").exists() else "❌"
        print(f"  {status} {m}")
else:
    print(f"❌ Restore path not found: {variants_input}")

RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Full model checkpoint
FULL_MODEL_CHECKPOINT = Path("/kaggle/input/datasets/aradhyasingh16/deepfake-ablation-checkpoints/ablation_study/checkpoints/full/model_best.pt")

# ──── SPEED SETTINGS ────
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS = 4
MAX_SAMPLES_PER_CLASS = 10000       # for training
BENCHMARK_TEST_SAMPLES = 1500       # per class for benchmark (3K total, balanced)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Degradation conditions
JPEG_QUALITIES = [100, 75, 50, 25]
RESIZE_SCALES = [0.75, 0.5]
CROP_RATIOS = [0.75, 0.5]

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──── Verify data ────
print(f"Device: {DEVICE}")
print(f"Data root: {DATA_ROOT}")
print(f"Exists: {DATA_ROOT.exists()}")
if not DATA_ROOT.exists():
    alt = Path("/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake")
    if alt.exists():
        DATA_ROOT = alt
        print(f"Using alternate path: {DATA_ROOT}")
    else:
        print("❌ DATA NOT FOUND! Run: !ls /kaggle/input/")

if DATA_ROOT.exists():
    for split in ["train", "valid", "test"]:
        sd = DATA_ROOT / split
        if sd.exists():
            rc = len(list((sd / "real").glob("*"))) if (sd / "real").exists() else 0
            fc = len(list((sd / "fake").glob("*"))) if (sd / "fake").exists() else 0
            print(f"  {split}: {rc} real, {fc} fake")

GLOBAL_START = time.time()
def elapsed():
    return f"[{(time.time()-GLOBAL_START)/3600:.1f}h elapsed]"

# %% [markdown]
# ## Cell 3: Model Architecture

# %%
class SpatialBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

class FrequencyBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
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
    def __init__(self, mode="full", feature_dim=2048, dropout_rate=0.5, pretrained=True):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim
        if mode in ("full", "fusion_no_attention", "spatial_only"):
            self.spatial_branch = SpatialBranch(pretrained=pretrained)
        if mode in ("full", "fusion_no_attention", "frequency_only"):
            self.frequency_branch = FrequencyBranch(pretrained=pretrained)
        if mode == "full":
            self.cbam = CBAM(in_features=feature_dim * 2)
        cin = feature_dim if mode in ("spatial_only", "frequency_only") else feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(cin, 512), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, spatial_input, frequency_input=None, return_freq_features=False):
        freq_features = None
        if self.mode == "spatial_only":
            features = self.spatial_branch(spatial_input)
        elif self.mode == "frequency_only":
            freq_features = self.frequency_branch(frequency_input)
            features = freq_features
        elif self.mode == "fusion_no_attention":
            s = self.spatial_branch(spatial_input)
            freq_features = self.frequency_branch(frequency_input)
            features = torch.cat([s, freq_features], dim=1)
        elif self.mode == "full":
            s = self.spatial_branch(spatial_input)
            freq_features = self.frequency_branch(frequency_input)
            features = self.cbam(torch.cat([s, freq_features], dim=1))
        prediction = self.classifier(features)
        if return_freq_features and freq_features is not None:
            return prediction, freq_features
        return prediction

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

print("✅ Model architecture defined")

# %% [markdown]
# ## Cell 4: Data Pipeline

# %%
class DCTTransform:
    def __init__(self, channels="rgb"):
        self.channels = channels
    def __call__(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image, dtype=np.float32)
        elif isinstance(image, torch.Tensor):
            img_array = image.permute(1, 2, 0).numpy() * 255.0
        else:
            img_array = image.astype(np.float32)
        dct_channels = []
        for c in range(3):
            ch = img_array[:, :, c] if len(img_array.shape) == 3 else img_array
            dc = dctn(ch, type=2, norm="ortho")
            dl = np.sign(dc) * np.log1p(np.abs(dc))
            dn = (dl - dl.min()) / (dl.max() - dl.min() + 1e-8)
            dct_channels.append(dn)
        return torch.from_numpy(np.stack(dct_channels, axis=0)).float()

class JPEGCompress:
    def __init__(self, quality=75):
        self.quality = quality
    def __call__(self, image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class ResizeDegradation:
    def __init__(self, scale=0.5, target_size=256):
        self.scale = scale
        self.target_size = target_size
    def __call__(self, image):
        w, h = image.size
        small = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        return image.resize(small, Image.BILINEAR).resize((self.target_size, self.target_size), Image.BILINEAR)

class CropDegradation:
    def __init__(self, crop_ratio=0.75, target_size=256):
        self.crop_ratio = crop_ratio
        self.target_size = target_size
    def __call__(self, image):
        w, h = image.size
        cw, ch = int(w * self.crop_ratio), int(h * self.crop_ratio)
        l, t = (w - cw) // 2, (h - ch) // 2
        return image.crop((l, t, l + cw, t + ch)).resize((self.target_size, self.target_size), Image.BILINEAR)

class DualStreamDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=256, max_samples=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Not found: {class_dir}")
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            if max_samples is not None:
                files = files[:max_samples]
            for p in files:
                self.samples.append((p, label_idx))
        self.spatial_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dct_tf = DCTTransform(channels="rgb")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.split == "train" and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.spatial_tf(img), self.dct_tf(img), torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self):
        labels = [s[1] for s in self.samples]
        return {"real": labels.count(0), "fake": labels.count(1), "total": len(labels)}


def get_dataloaders(data_root, image_size=256, batch_size=32, num_workers=4, max_samples=None):
    loaders = {}
    for split in ("train", "valid", "test"):
        sd = Path(data_root) / split
        if not sd.exists():
            print(f"  [!] Skipping missing split: {sd}")
            continue
        ds = DualStreamDataset(sd, split=split, image_size=image_size, max_samples=max_samples)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        )
        c = ds.get_class_counts()
        print(f"  {split:5s} -> {c['total']:,} images (real: {c['real']:,}, fake: {c['fake']:,}) | {len(loaders[split])} batches")
    return loaders

print("✅ Data pipeline defined")

# %% [markdown]
# ## Cell 5: Training Engine

# %%
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.should_stop = False
    def __call__(self, val):
        if self.best is None or val < self.best:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def _safe_auc(labels, preds):
    try: return roc_auc_score(labels, preds)
    except: return 0.0

class Trainer:
    def __init__(self, model, train_loader, val_loader, *, device="cuda",
                 lr=1e-4, checkpoint_dir="checkpoints", results_dir="results"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss, n = 0.0, 0
        all_labels, all_preds = [], []
        for batch_idx, (spatial, freq, labels) in enumerate(self.train_loader):
            spatial, freq = spatial.to(self.device), freq.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            pred = self.model(spatial, freq)
            loss = self.criterion(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item(); n += 1
            all_labels.extend(labels.detach().cpu().squeeze(1).tolist())
            all_preds.extend(pred.detach().cpu().squeeze(1).tolist())
            if (batch_idx + 1) % 200 == 0:
                print(f"    Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {running_loss/n:.4f} {elapsed()}")
        bp = [1.0 if p > 0.5 else 0.0 for p in all_preds]
        return {"train_loss": running_loss/max(n,1), "train_acc": accuracy_score(all_labels, bp),
                "train_auc": _safe_auc(all_labels, all_preds)}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        running_loss, n = 0.0, 0
        all_labels, all_preds = [], []
        for spatial, freq, labels in self.val_loader:
            spatial, freq = spatial.to(self.device), freq.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            pred = self.model(spatial, freq)
            loss = self.criterion(pred, labels)
            running_loss += loss.item(); n += 1
            all_labels.extend(labels.cpu().squeeze(1).tolist())
            all_preds.extend(pred.cpu().squeeze(1).tolist())
        bp = [1.0 if p > 0.5 else 0.0 for p in all_preds]
        return {"val_loss": running_loss/max(n,1), "val_acc": accuracy_score(all_labels, bp),
                "val_auc": _safe_auc(all_labels, all_preds),
                "val_f1": f1_score(all_labels, bp, zero_division=0)}

    def save_checkpoint(self, epoch, metrics, tag="best"):
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict(),
                     "metrics": metrics, "model_mode": self.model.mode}, path)
        print(f"    [SAVE] {path.name}")

    def fit(self, num_epochs=10, patience=3):
        es = EarlyStopping(patience=patience)
        best_val_loss = float("inf")
        mode = getattr(self.model, "mode", "?")
        print(f"\n{'='*60}")
        print(f"  Training: {mode} | {num_epochs} epochs | patience={patience}")
        print(f"  Train: {len(self.train_loader)} batches | Val: {len(self.val_loader)} batches")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            tm = self._train_one_epoch(epoch)
            vm = self._validate()
            em = {"epoch": epoch, **tm, **vm, "time_s": time.time() - t0}
            self.history.append(em)
            print(f"  Epoch {epoch}/{num_epochs} | TrLoss: {tm['train_loss']:.4f} TrAcc: {tm['train_acc']:.4f} | "
                  f"VLoss: {vm['val_loss']:.4f} VAcc: {vm['val_acc']:.4f} VAUC: {vm['val_auc']:.4f} | "
                  f"{em['time_s']:.0f}s {elapsed()}")
            self.scheduler.step(vm["val_loss"])
            if vm["val_loss"] < best_val_loss:
                best_val_loss = vm["val_loss"]
                self.save_checkpoint(epoch, em, tag="best")
            if es(vm["val_loss"]):
                print(f"  [STOP] Early stopping at epoch {epoch}")
                break

        self.save_checkpoint(epoch, em, tag="last")
        best = min(self.history, key=lambda h: h["val_loss"])
        print(f"  ✅ Done! Best: epoch {best['epoch']} | VAcc: {best['val_acc']:.4f} | VAUC: {best['val_auc']:.4f}")

        with open(self.results_dir / "training_summary.json", "w") as f:
            json.dump({"best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
                        "best_val_auc": best["val_auc"], "mode": mode}, f, indent=2)
        return self.history

print("✅ Training engine defined")

# %% [markdown]
# ## Cell 6: Load Data

# %%
print(f"[*] Loading datasets (max {MAX_SAMPLES_PER_CLASS} per class for train/val)... {elapsed()}")
loaders = get_dataloaders(
    data_root=DATA_ROOT, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, max_samples=MAX_SAMPLES_PER_CLASS,
)
assert "train" in loaders and "valid" in loaders, "Missing train/valid splits!"
print(f"\n✅ Data loaded {elapsed()}")

# %% [markdown]
# ## Cell 7: Train ALL 4 Variants

# %%
if FULL_MODEL_CHECKPOINT is not None and Path(FULL_MODEL_CHECKPOINT).exists():
    ABLATION_MODES = ["spatial_only", "frequency_only", "fusion_no_attention"]
    print(f"✅ Full model checkpoint found — training 3 variants only")
else:
    ABLATION_MODES = ["full", "spatial_only", "frequency_only", "fusion_no_attention"]
    print(f"⚠️ No checkpoint — training ALL 4 variants")

ablation_summaries = {}

for mode in ABLATION_MODES:
    print(f"\n{'#'*70}")
    print(f"#  TRAINING: {mode.upper()}  {elapsed()}")
    print(f"{'#'*70}")

    ckpt_dir = CHECKPOINT_DIR / mode
    res_dir = RESULTS_DIR / mode

    if (ckpt_dir / "model_best.pt").exists():
        print(f"  [SKIP] Already trained — loading checkpoint")
        ckpt = torch.load(ckpt_dir / "model_best.pt", map_location="cpu", weights_only=False)
        ablation_summaries[mode] = ckpt.get("metrics", {})
        continue

    model = HybridDetector(mode=mode, feature_dim=2048, dropout_rate=0.5, pretrained=True)
    params = model.get_num_parameters()
    print(f"  Params: {params['total']:,} total, {params['trainable']:,} trainable")

    trainer = Trainer(model, loaders["train"], loaders["valid"],
                      device=DEVICE, lr=LEARNING_RATE,
                      checkpoint_dir=str(ckpt_dir), results_dir=str(res_dir))
    history = trainer.fit(num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)

    best = min(history, key=lambda h: h["val_loss"])
    ablation_summaries[mode] = best
    del model, trainer
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"  ALL VARIANTS TRAINED! {elapsed()}")
print(f"{'='*70}")
for m, metrics in ablation_summaries.items():
    acc = metrics.get("val_acc", "?")
    auc = metrics.get("val_auc", "?")
    if isinstance(acc, float): acc = f"{acc:.4f}"
    if isinstance(auc, float): auc = f"{auc:.4f}"
    print(f"  {m:25s} | Val Acc: {acc} | Val AUC: {auc}")

# %% [markdown]
# ## Cell 8: Degradation Benchmark Engine

# %%
class DegradationBenchmark:
    def __init__(self, model, test_dataset, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = test_dataset
        self.device = device
        self.dct_tf = DCTTransform(channels="rgb")
        self.spatial_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _compute_metrics(self, labels, preds):
        bp = [1.0 if p > 0.5 else 0.0 for p in preds]
        return {
            "accuracy": accuracy_score(labels, bp) * 100,
            "auc": _safe_auc(labels, preds),
            "f1": f1_score(labels, bp, zero_division=0),
        }

    @torch.no_grad()
    def evaluate_clean(self, loader):
        all_labels, all_preds = [], []
        for spatial, freq, labels in loader:
            pred = self.model(spatial.to(self.device), freq.to(self.device))
            all_labels.extend(labels.tolist())
            all_preds.extend(pred.cpu().squeeze(1).tolist())
        return self._compute_metrics(all_labels, all_preds)

    @torch.no_grad()
    def evaluate_degraded(self, degradation_fn):
        """Evaluate on ALL images in the dataset (no max_images truncation)."""
        all_labels, all_preds = [], []
        n = len(self.dataset)
        for idx in range(n):
            path, label = self.dataset.samples[idx]
            img = Image.open(path).convert("RGB").resize(
                (self.dataset.image_size, self.dataset.image_size), Image.BILINEAR)
            degraded = degradation_fn(img)
            spatial = self.spatial_tf(degraded).unsqueeze(0).to(self.device)
            freq = self.dct_tf(degraded).unsqueeze(0).to(self.device)
            pred = self.model(spatial, freq)
            all_labels.append(label)
            all_preds.append(pred.cpu().squeeze().item())
            if (idx + 1) % 1000 == 0:
                print(f"      {idx+1}/{n} images...")
        return self._compute_metrics(all_labels, all_preds)

    def run_benchmark(self, loader):
        results = {}
        print("    ▸ Clean...")
        results["Clean"] = self.evaluate_clean(loader)

        for q in JPEG_QUALITIES:
            k = f"JPEG Q={q}"
            print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(JPEGCompress(quality=q))

        for s in RESIZE_SCALES:
            k = f"Resize {int(s*100)}%"
            print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(ResizeDegradation(scale=s, target_size=IMAGE_SIZE))

        for r in CROP_RATIOS:
            k = f"Crop {int(r*100)}%"
            print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(CropDegradation(crop_ratio=r, target_size=IMAGE_SIZE))

        print("    ▸ Combined (Q50+Resize75%)...")
        def combined(img):
            return ResizeDegradation(0.75, IMAGE_SIZE)(JPEGCompress(50)(img))
        results["Combined (Q50+Res75%)"] = self.evaluate_degraded(combined)

        return results

print("✅ Benchmark engine defined")

# %% [markdown]
# ## Cell 9: Run Benchmarks on All 4 Variants
#
# **FIX APPLIED HERE**: Build a separate balanced test dataset with
# `max_samples=BENCHMARK_TEST_SAMPLES` per class (1500 real + 1500 fake = 3000).
# `evaluate_degraded` now iterates over ALL samples in this dataset, guaranteeing
# both classes are always present.

# %%
ALL_MODES = ["full", "spatial_only", "frequency_only", "fusion_no_attention"]
all_benchmark_results = {}

# ──────────────────────────────────────────────────────────────────────
# FIX: Build a SEPARATE balanced test dataset for benchmarking
# Previously used the training test loader (10K real + 10K fake) but
# evaluate_degraded only iterated first 4K samples → all real.
# Now: 1500 real + 1500 fake = 3000 total, iterate ALL of them.
# ──────────────────────────────────────────────────────────────────────
test_dir = DATA_ROOT / "test"
bench_test_ds = DualStreamDataset(
    test_dir, split="test", image_size=IMAGE_SIZE,
    max_samples=BENCHMARK_TEST_SAMPLES  # 1500 per class → 3000 total balanced
)
bench_test_loader = DataLoader(
    bench_test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ──── Sanity check: verify balance ────
bench_counts = bench_test_ds.get_class_counts()
print(f"\n{'='*60}")
print(f"  BENCHMARK TEST SET — BALANCE CHECK")
print(f"{'='*60}")
print(f"  Real:  {bench_counts['real']}")
print(f"  Fake:  {bench_counts['fake']}")
print(f"  Total: {bench_counts['total']}")
assert bench_counts['real'] > 0 and bench_counts['fake'] > 0, \
    f"❌ TEST SET IS NOT BALANCED! real={bench_counts['real']}, fake={bench_counts['fake']}"
assert bench_counts['real'] == bench_counts['fake'], \
    f"⚠️ Unequal classes: real={bench_counts['real']}, fake={bench_counts['fake']}"
print(f"  ✅ Balanced: {bench_counts['real']} per class")
print(f"\nStarting benchmarks... {elapsed()}\n")

for mode in ALL_MODES:
    print(f"{'='*60}")
    print(f"  BENCHMARKING: {mode.upper()}  {elapsed()}")
    print(f"{'='*60}")

    model = HybridDetector(mode=mode, feature_dim=2048, dropout_rate=0.5, pretrained=False)

    # Load checkpoint
    if mode == "full" and FULL_MODEL_CHECKPOINT is not None and Path(FULL_MODEL_CHECKPOINT).exists():
        ckpt = torch.load(FULL_MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        remapped = {k.replace("cbam.channel_attention.", "cbam.ca."): v for k, v in state_dict.items()}
        model.load_state_dict(remapped)
        print(f"  ✅ Loaded full model from external checkpoint (with key remapping)")
    else:
        ckpt_path = CHECKPOINT_DIR / mode / "model_best.pt"
        if not ckpt_path.exists():
            print(f"  ❌ No checkpoint for {mode} — skipping")
            del model; continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✅ Loaded from: {ckpt_path.name}")

    # Use the balanced benchmark dataset and loader
    bench = DegradationBenchmark(model, bench_test_ds, device=DEVICE)
    results = bench.run_benchmark(bench_test_loader)
    all_benchmark_results[mode] = results

    # Print summary with AUC to verify it's not NaN
    clean_acc = results["Clean"]["accuracy"]
    print(f"\n  {'Condition':<28s} {'Acc%':>7s} {'AUC':>7s} {'F1':>7s} {'Drop':>8s}")
    print(f"  {'-'*62}")
    for cond, m in results.items():
        drop = f"({m['accuracy']-clean_acc:+.1f})" if cond != "Clean" else ""
        auc_str = f"{m['auc']:.4f}" if not (m['auc'] != m['auc']) else "NaN!!"  # NaN check
        print(f"  {cond:<28s} {m['accuracy']:>6.1f}% {auc_str:>7s} {m['f1']:>7.4f} {drop:>8s}")

    # Verify no NaN AUCs (would indicate single-class bug)
    for cond, m in results.items():
        if m['auc'] != m['auc']:  # NaN check
            print(f"  ⚠️ WARNING: NaN AUC at {cond} — test set may still be imbalanced!")

    del model, bench
    torch.cuda.empty_cache()

# Save
with open(RESULTS_DIR / "all_benchmark_results.json", "w") as f:
    json.dump(all_benchmark_results, f, indent=2)
print(f"\n✅ All benchmarks done! {elapsed()}")

# %% [markdown]
# ## Cell 10: Results Table

# %%
mode_labels = {
    "full": "S+F+CBAM",
    "spatial_only": "S-only",
    "frequency_only": "F-only",
    "fusion_no_attention": "S+F (No Attn)",
}

modes_present = [m for m in ALL_MODES if m in all_benchmark_results]
all_conditions = []
for mr in all_benchmark_results.values():
    for c in mr:
        if c not in all_conditions:
            all_conditions.append(c)

print("\n" + "=" * 90)
print("  ABLATION STUDY v3 — COMPLETE RESULTS (FIXED BALANCED BENCHMARK)")
print("=" * 90)

header = f"{'Condition':<28s}"
for m in modes_present:
    header += f" | {mode_labels[m]:>14s}"
print(header)
print("-" * len(header))

for cond in all_conditions:
    row = f"{cond:<28s}"
    for m in modes_present:
        if cond in all_benchmark_results[m]:
            row += f" | {all_benchmark_results[m][cond]['accuracy']:>13.1f}%"
        else:
            row += f" | {'N/A':>14s}"
    print(row)

# Drop table
print(f"\n  ACCURACY DROP FROM CLEAN (pp):")
print("-" * 80)
for cond in all_conditions:
    if cond == "Clean": continue
    row = f"{cond:<28s}"
    for m in modes_present:
        clean = all_benchmark_results[m]["Clean"]["accuracy"]
        if cond in all_benchmark_results[m]:
            drop = all_benchmark_results[m][cond]["accuracy"] - clean
            row += f" | {drop:>+13.1f}%"
        else:
            row += f" | {'N/A':>14s}"
    print(row)

# %% [markdown]
# ## Cell 11: Visualization Plots

# %%
colors = {"full": "#2196F3", "spatial_only": "#4CAF50", "frequency_only": "#F44336", "fusion_no_attention": "#FF9800"}
markers = {"full": "o", "spatial_only": "s", "frequency_only": "^", "fusion_no_attention": "D"}

# ────── Plot 1: JPEG Compression vs Accuracy ──────
jpeg_conds = ["Clean"] + [f"JPEG Q={q}" for q in JPEG_QUALITIES]
jpeg_labels = ["Clean", "Q=100", "Q=75", "Q=50", "Q=25"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Ablation Study: JPEG Compression Impact (Fixed Balanced Benchmark)", fontsize=14, fontweight="bold")

for ax_idx, (metric, ylabel) in enumerate([("accuracy", "Accuracy (%)"), ("auc", "AUC-ROC"), ("f1", "F1 Score")]):
    ax = axes[ax_idx]
    for m in modes_present:
        vals = [all_benchmark_results[m].get(c, {}).get(metric) for c in jpeg_conds]
        vx = [i for i, v in enumerate(vals) if v is not None]
        vy = [v for v in vals if v is not None]
        ax.plot(vx, vy, f"-{markers[m]}", color=colors[m], label=mode_labels[m], linewidth=2, markersize=8)
    ax.set_xticks(range(len(jpeg_labels)))
    ax.set_xticklabels(jpeg_labels, rotation=45)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "jpeg_compression_ablation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved jpeg_compression_ablation.png")

# ────── Plot 2: Accuracy Drop Heatmap ──────
conds_no_clean = [c for c in all_conditions if c != "Clean"]
drop_matrix = []
for m in modes_present:
    clean = all_benchmark_results[m]["Clean"]["accuracy"]
    row = [all_benchmark_results[m].get(c, {}).get("accuracy", clean) - clean for c in conds_no_clean]
    drop_matrix.append(row)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(np.array(drop_matrix), annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            vmax=0, xticklabels=conds_no_clean, yticklabels=[mode_labels[m] for m in modes_present],
            ax=ax, linewidths=0.5, annot_kws={"fontsize": 9})
ax.set_title("Accuracy Drop from Clean (pp) — More Negative = More Vulnerable", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy_drop_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved accuracy_drop_heatmap.png")

# ────── Plot 3: Grouped Bar Chart ──────
key_conds = [c for c in ["Clean", "JPEG Q=75", "JPEG Q=50", "JPEG Q=25", "Resize 50%", "Combined (Q50+Res75%)"]
             if any(c in all_benchmark_results.get(m, {}) for m in modes_present)]
x = np.arange(len(key_conds))
width = 0.2

fig, ax = plt.subplots(figsize=(16, 7))
for i, m in enumerate(modes_present):
    accs = [all_benchmark_results[m].get(c, {}).get("accuracy", 0) for c in key_conds]
    offset = (i - len(modes_present)/2 + 0.5) * width
    bars = ax.bar(x + offset, accs, width, label=mode_labels[m], color=colors[m], alpha=0.85)
    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Ablation Study: Accuracy Under Key Degradations", fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(key_conds, rotation=30, ha="right")
ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim([0, 110])
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grouped_bar_ablation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved grouped_bar_ablation.png")

# ────── Plot 4: Spatial vs Frequency Direct Comparison ──────
if "spatial_only" in all_benchmark_results and "frequency_only" in all_benchmark_results:
    common = [c for c in all_conditions if c != "Clean"
              and c in all_benchmark_results["spatial_only"] and c in all_benchmark_results["frequency_only"]]
    sc = all_benchmark_results["spatial_only"]["Clean"]["accuracy"]
    fc = all_benchmark_results["frequency_only"]["Clean"]["accuracy"]
    s_drops = [all_benchmark_results["spatial_only"][c]["accuracy"] - sc for c in common]
    f_drops = [all_benchmark_results["frequency_only"][c]["accuracy"] - fc for c in common]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common)); w = 0.35
    ax.bar(x - w/2, s_drops, w, label="Spatial-Only", color=colors["spatial_only"], alpha=0.85)
    ax.bar(x + w/2, f_drops, w, label="Frequency-Only", color=colors["frequency_only"], alpha=0.85)
    for xi, (sv, fv) in enumerate(zip(s_drops, f_drops)):
        ax.text(xi - w/2, sv - 1.5, f"{sv:.1f}", ha="center", fontsize=8, color="white", fontweight="bold")
        ax.text(xi + w/2, fv - 1.5, f"{fv:.1f}", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_title("Spatial vs Frequency: Which Branch Breaks More?\n(More negative = more vulnerable)", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=45, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "spatial_vs_frequency_drops.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved spatial_vs_frequency_drops.png")

# ────── Plot 5: CBAM Benefit ──────
if "full" in all_benchmark_results and "fusion_no_attention" in all_benchmark_results:
    common = [c for c in all_conditions if c in all_benchmark_results["full"] and c in all_benchmark_results["fusion_no_attention"]]
    benefit = [all_benchmark_results["full"][c]["accuracy"] - all_benchmark_results["fusion_no_attention"][c]["accuracy"] for c in common]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common))
    ax.bar(x, benefit, color=["#2196F3" if b >= 0 else "#F44336" for b in benefit], alpha=0.85, edgecolor="black", linewidth=0.5)
    for xi, val in enumerate(benefit):
        ax.text(xi, val + (0.3 if val >= 0 else -0.8), f"{val:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Accuracy Gain from CBAM (pp)")
    ax.set_title("CBAM Benefit: S+F+CBAM minus S+F\n(Positive = CBAM helps)", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linewidth=1); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cbam_benefit.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved cbam_benefit.png")

# %% [markdown]
# ## Cell 12: Key Findings

# %%
print(f"\n{'='*70}")
print(f"  🔬 ABLATION STUDY v3 — KEY FINDINGS (FIXED)  {elapsed()}")
print(f"{'='*70}")

if "spatial_only" in all_benchmark_results and "frequency_only" in all_benchmark_results:
    print("\n📊 1. Clean Accuracy:")
    for m in modes_present:
        print(f"    {mode_labels[m]:>16s}: {all_benchmark_results[m]['Clean']['accuracy']:.1f}%")

    print("\n📊 2. JPEG Q=25 (worst compression):")
    for m in modes_present:
        if "JPEG Q=25" in all_benchmark_results[m]:
            acc = all_benchmark_results[m]["JPEG Q=25"]["accuracy"]
            clean = all_benchmark_results[m]["Clean"]["accuracy"]
            print(f"    {mode_labels[m]:>16s}: {acc:.1f}% (drop: {acc-clean:+.1f}pp)")

    print("\n📊 3. Branch Vulnerability (JPEG Q=50):")
    if "JPEG Q=50" in all_benchmark_results["spatial_only"] and "JPEG Q=50" in all_benchmark_results["frequency_only"]:
        sc = all_benchmark_results["spatial_only"]["Clean"]["accuracy"]
        fc = all_benchmark_results["frequency_only"]["Clean"]["accuracy"]
        sd = all_benchmark_results["spatial_only"]["JPEG Q=50"]["accuracy"] - sc
        fd = all_benchmark_results["frequency_only"]["JPEG Q=50"]["accuracy"] - fc
        print(f"    Spatial drop:   {sd:+.1f}pp")
        print(f"    Frequency drop: {fd:+.1f}pp")
        if abs(fd) > abs(sd):
            print(f"\n    ✅ CONFIRMED: Frequency branch is {abs(fd)/max(abs(sd),0.1):.1f}× MORE vulnerable")
            print(f"    → Consistency training should target the frequency branch")
        else:
            print(f"\n    ⚠️ Spatial branch drops more — interesting finding for the paper")

    print(f"\n📊 4. Average JPEG Drop:")
    for m in modes_present:
        clean = all_benchmark_results[m]["Clean"]["accuracy"]
        drops = [all_benchmark_results[m][f"JPEG Q={q}"]["accuracy"] - clean
                 for q in JPEG_QUALITIES if f"JPEG Q={q}" in all_benchmark_results[m]]
        if drops:
            print(f"    {mode_labels[m]:>16s}: {np.mean(drops):+.1f}pp avg")

    # Verify AUC sanity
    print(f"\n📊 5. AUC Sanity Check (should all be valid numbers, not NaN):")
    all_ok = True
    for m in modes_present:
        for cond in all_conditions:
            auc = all_benchmark_results[m].get(cond, {}).get("auc", 0)
            if auc != auc:  # NaN check
                print(f"    ❌ NaN AUC: {mode_labels[m]} @ {cond}")
                all_ok = False
    if all_ok:
        print(f"    ✅ All AUCs are valid numbers — benchmark is balanced correctly")

# Save final results
final = {
    "experiment": "ablation_study_v3_fixed", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "fix_applied": "Built separate balanced test dataset (1500 per class) for benchmark. "
                   "v2 bug: evaluate_degraded took first 4K of 20K samples (all real).",
    "config": {"image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
               "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
               "benchmark_test_samples_per_class": BENCHMARK_TEST_SAMPLES,
               "benchmark_total_images": bench_counts["total"]},
    "clean_accuracy": {m: all_benchmark_results[m]["Clean"]["accuracy"] for m in modes_present},
    "benchmark_results": all_benchmark_results,
}
with open(RESULTS_DIR / "ablation_study_complete.json", "w") as f:
    json.dump(final, f, indent=2)

print(f"\n{'='*70}")
print(f"  ✅ ABLATION STUDY v3 COMPLETE (FIXED)! {elapsed()}")
print(f"  Next step: Consistency training on the weak branch")
print(f"{'='*70}")
