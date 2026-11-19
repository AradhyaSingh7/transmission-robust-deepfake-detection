#!/usr/bin/env python
# ============================================================================
# CONSISTENCY TRAINING — Cosine Similarity Loss — Kaggle Notebook 2/2
# ============================================================================
# Fine-tunes the full model (S+F+CBAM) with consistency regularization
# using Cosine Distance loss (1 - cos_sim) between clean and JPEG-compressed
# branch feature representations.
#
# Models trained (7 total):
#   1. Naive Augmentation (random JPEG aug, BCE only)
#   2-4. Spatial Consistency Cosine λ = [0.01, 0.1, 1.0]
#   5-7. Dual-Branch Consistency Cosine λ = [0.01, 0.1, 1.0]
#
# Then benchmarks all 7 + original baseline under degradation.
#
# Estimated time: ~7-8 hours on T4 GPU
# ============================================================================

# %% [markdown]
# # 🛠️ Consistency Training (Cosine Similarity Loss)

# %% [markdown]
# ## Cell 1: Setup

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])

# %% [markdown]
# ## Cell 2: Imports & Config

# %%
import os, json, time, random, io, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION — Update these paths for your Kaggle setup
# ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake")
FULL_MODEL_CHECKPOINT = Path("/kaggle/input/datasets/aradhyasingh16/deepfake-ablation-checkpoints/ablation_study/checkpoints/full/model_best.pt")

OUTPUT_DIR = Path("/kaggle/working/consistency_cosine")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ──── SPEED SETTINGS ────
IMAGE_SIZE = 256
BATCH_SIZE = 32
FINE_TUNE_LR = 1e-5
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
NUM_WORKERS = 4
MAX_SAMPLES_PER_CLASS = 5000
BENCHMARK_TEST_SAMPLES = 1500  # per class (3K total)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_VALUES = [0.01, 0.1, 1.0]
JPEG_QUALITY_RANGE = (25, 75)
JPEG_QUALITIES = [100, 75, 50, 25]
RESIZE_SCALES = [0.75, 0.5]
CROP_RATIOS = [0.75, 0.5]

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
assert DEVICE == "cuda", "⚠️ GPU not detected! Enable GPU in notebook settings."
print(f"Data root exists: {DATA_ROOT.exists()}")
print(f"Checkpoint exists: {FULL_MODEL_CHECKPOINT.exists()}")

GLOBAL_START = time.time()
def elapsed():
    return f"[{(time.time()-GLOBAL_START)/3600:.1f}h]"

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

    def forward(self, spatial_input, frequency_input=None):
        if self.mode == "full":
            s = self.spatial_branch(spatial_input)
            f = self.frequency_branch(frequency_input)
            features = self.cbam(torch.cat([s, f], dim=1))
        elif self.mode == "spatial_only":
            features = self.spatial_branch(spatial_input)
        elif self.mode == "frequency_only":
            features = self.frequency_branch(frequency_input)
        elif self.mode == "fusion_no_attention":
            s = self.spatial_branch(spatial_input)
            f = self.frequency_branch(frequency_input)
            features = torch.cat([s, f], dim=1)
        return self.classifier(features)

print("✅ Model architecture defined")

# %% [markdown]
# ## Cell 4: Data Pipeline

# %%
class DCTTransform:
    def __init__(self):
        pass
    def __call__(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image, dtype=np.float32)
        else:
            img_array = image.astype(np.float32)
        dct_channels = []
        for c in range(3):
            ch = img_array[:, :, c]
            dc = dctn(ch, type=2, norm="ortho")
            dl = np.sign(dc) * np.log1p(np.abs(dc))
            dn = (dl - dl.min()) / (dl.max() - dl.min() + 1e-8)
            dct_channels.append(dn)
        return torch.from_numpy(np.stack(dct_channels, axis=0)).float()

def jpeg_compress_pil(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

SPATIAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
DCT_TF = DCTTransform()

class DualStreamDataset(Dataset):
    """Standard dataset for validation/test/benchmark."""
    def __init__(self, root_dir, split="train", image_size=256, max_samples=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = self.root_dir / label_name
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            if max_samples:
                files = files[:max_samples]
            for p in files:
                self.samples.append((p, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.split == "train" and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return SPATIAL_TF(img), DCT_TF(img), torch.tensor(label, dtype=torch.float32)

class AugmentedDataset(Dataset):
    """For naive augmentation: randomly applies JPEG compression."""
    def __init__(self, root_dir, image_size=256, max_samples=None,
                 aug_prob=0.5, jpeg_range=(25, 75)):
        self.image_size = image_size
        self.aug_prob = aug_prob
        self.jpeg_range = jpeg_range
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = Path(root_dir) / label_name
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            if max_samples:
                files = files[:max_samples]
            for p in files:
                self.samples.append((p, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if random.random() < self.aug_prob:
            img = jpeg_compress_pil(img, random.randint(*self.jpeg_range))
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return SPATIAL_TF(img), DCT_TF(img), torch.tensor(label, dtype=torch.float32)

class ConsistencyDataset(Dataset):
    """Returns clean + JPEG-compressed pairs for consistency training."""
    def __init__(self, root_dir, image_size=256, max_samples=None,
                 jpeg_range=(25, 75)):
        self.image_size = image_size
        self.jpeg_range = jpeg_range
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = Path(root_dir) / label_name
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            if max_samples:
                files = files[:max_samples]
            for p in files:
                self.samples.append((p, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        compressed = jpeg_compress_pil(img, random.randint(*self.jpeg_range))
        # Apply same flip to both
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            compressed = compressed.transpose(Image.FLIP_LEFT_RIGHT)
        return (SPATIAL_TF(img), DCT_TF(img),
                SPATIAL_TF(compressed), DCT_TF(compressed),
                torch.tensor(label, dtype=torch.float32))

print("✅ Data pipeline defined")

# %% [markdown]
# ## Cell 5: Trainers

# %%
def _safe_auc(labels, preds):
    try: return roc_auc_score(labels, preds)
    except: return 0.0

class EarlyStopping:
    def __init__(self, patience=2):
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

class NaiveAugTrainer:
    """Standard BCE trainer for naive augmentation baseline."""
    def __init__(self, model, train_loader, val_loader, *, device="cuda",
                 lr=1e-5, checkpoint_dir="ckpt", results_dir="res"):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=1)
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir = Path(results_dir); self.res_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _train_epoch(self):
        self.model.train()
        loss_sum, n, labels_all, preds_all = 0, 0, [], []
        for i, (s, f, lab) in enumerate(self.train_loader):
            s, f, lab = s.to(self.device), f.to(self.device), lab.to(self.device).unsqueeze(1)
            pred = self.model(s, f)
            loss = self.criterion(pred, lab)
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
            loss_sum += loss.item(); n += 1
            labels_all.extend(lab.detach().cpu().squeeze(1).tolist())
            preds_all.extend(pred.detach().cpu().squeeze(1).tolist())
            if (i+1) % 100 == 0: print(f"    Batch {i+1}/{len(self.train_loader)} | Loss: {loss_sum/n:.4f} {elapsed()}")
        bp = [1 if p > 0.5 else 0 for p in preds_all]
        return {"train_loss": loss_sum/max(n,1), "train_acc": accuracy_score(labels_all, bp)}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_sum, n, labels_all, preds_all = 0, 0, [], []
        for s, f, lab in self.val_loader:
            s, f, lab = s.to(self.device), f.to(self.device), lab.to(self.device).unsqueeze(1)
            pred = self.model(s, f)
            loss_sum += self.criterion(pred, lab).item(); n += 1
            labels_all.extend(lab.cpu().squeeze(1).tolist())
            preds_all.extend(pred.cpu().squeeze(1).tolist())
        bp = [1 if p > 0.5 else 0 for p in preds_all]
        return {"val_loss": loss_sum/max(n,1), "val_acc": accuracy_score(labels_all, bp),
                "val_auc": _safe_auc(labels_all, preds_all)}

    def fit(self, num_epochs=5, patience=2):
        es = EarlyStopping(patience); best_vl = float("inf")
        print(f"\n  Training naive_aug | {num_epochs} epochs | {len(self.train_loader)} batches")
        for ep in range(1, num_epochs+1):
            t0 = time.time()
            tm = self._train_epoch(); vm = self._validate()
            em = {"epoch": ep, **tm, **vm, "time_s": time.time()-t0}; self.history.append(em)
            print(f"  Ep {ep}/{num_epochs} | TrL: {tm['train_loss']:.4f} TrA: {tm['train_acc']:.4f} | "
                  f"VL: {vm['val_loss']:.4f} VA: {vm['val_acc']:.4f} | {em['time_s']:.0f}s {elapsed()}")
            self.scheduler.step(vm["val_loss"])
            if vm["val_loss"] < best_vl:
                best_vl = vm["val_loss"]
                torch.save({"model_state_dict": self.model.state_dict(), "metrics": em},
                           self.ckpt_dir / "model_best.pt")
            if es(vm["val_loss"]): print(f"  [STOP] Early stopping at epoch {ep}"); break
        best = min(self.history, key=lambda h: h["val_loss"])
        print(f"  ✅ Best: ep {best['epoch']} | VA: {best['val_acc']:.4f}")
        return self.history


class ConsistencyTrainer:
    """Trains with BCE + λ * (1 - CosineSimilarity) consistency loss on branch features."""
    def __init__(self, model, train_loader, val_loader, *,
                 target="spatial",  # "spatial" or "dual"
                 lambda_val=0.1, device="cuda", lr=1e-5,
                 checkpoint_dir="ckpt", results_dir="res"):
        self.model = model.to(device)
        self.device = device
        self.target = target
        self.lambda_val = lambda_val
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=1)
        self.ckpt_dir = Path(checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir = Path(results_dir); self.res_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _train_epoch(self):
        self.model.train()
        cls_sum, con_sum, n = 0, 0, 0
        labels_all, preds_all = [], []
        for i, (s_c, f_c, s_x, f_x, lab) in enumerate(self.train_loader):
            s_c = s_c.to(self.device); f_c = f_c.to(self.device)
            s_x = s_x.to(self.device); lab = lab.to(self.device).unsqueeze(1)

            # Extract branch features on clean input
            sf_c = self.model.spatial_branch(s_c)
            ff_c = self.model.frequency_branch(f_c)

            # Classification (manual forward to avoid recomputing branches)
            fused = torch.cat([sf_c, ff_c], dim=1)
            attended = self.model.cbam(fused)
            pred = self.model.classifier(attended)
            cls_loss = self.criterion(pred, lab)

            # Cosine distance consistency loss on spatial branch: 1 - cos_sim
            sf_x = self.model.spatial_branch(s_x)
            con_loss = 1.0 - F.cosine_similarity(sf_c, sf_x, dim=-1).mean()

            if self.target == "dual":
                f_x = f_x.to(self.device)
                ff_x = self.model.frequency_branch(f_x)
                dual_loss = 1.0 - F.cosine_similarity(ff_c, ff_x, dim=-1).mean()
                con_loss = (con_loss + dual_loss) / 2.0

            total_loss = cls_loss + self.lambda_val * con_loss
            self.optimizer.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()

            cls_sum += cls_loss.item(); con_sum += con_loss.item(); n += 1
            labels_all.extend(lab.detach().cpu().squeeze(1).tolist())
            preds_all.extend(pred.detach().cpu().squeeze(1).tolist())
            if (i+1) % 100 == 0:
                print(f"    Batch {i+1}/{len(self.train_loader)} | CLS: {cls_sum/n:.4f} CON: {con_sum/n:.4f} {elapsed()}")

        bp = [1 if p > 0.5 else 0 for p in preds_all]
        return {"cls_loss": cls_sum/max(n,1), "con_loss": con_sum/max(n,1),
                "train_acc": accuracy_score(labels_all, bp)}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_sum, n, labels_all, preds_all = 0, 0, [], []
        for s, f, lab in self.val_loader:
            s, f, lab = s.to(self.device), f.to(self.device), lab.to(self.device).unsqueeze(1)
            pred = self.model(s, f)
            loss_sum += self.criterion(pred, lab).item(); n += 1
            labels_all.extend(lab.cpu().squeeze(1).tolist())
            preds_all.extend(pred.cpu().squeeze(1).tolist())
        bp = [1 if p > 0.5 else 0 for p in preds_all]
        return {"val_loss": loss_sum/max(n,1), "val_acc": accuracy_score(labels_all, bp),
                "val_auc": _safe_auc(labels_all, preds_all)}

    def fit(self, num_epochs=5, patience=2):
        es = EarlyStopping(patience); best_vl = float("inf")
        tag = f"cosine_{self.target}_l{self.lambda_val}"
        print(f"\n  Training {tag} | {num_epochs} epochs | {len(self.train_loader)} batches")
        for ep in range(1, num_epochs+1):
            t0 = time.time()
            tm = self._train_epoch(); vm = self._validate()
            em = {"epoch": ep, **tm, **vm, "time_s": time.time()-t0}; self.history.append(em)
            print(f"  Ep {ep}/{num_epochs} | CLS: {tm['cls_loss']:.4f} CON: {tm['con_loss']:.4f} "
                  f"TrA: {tm['train_acc']:.4f} | VL: {vm['val_loss']:.4f} VA: {vm['val_acc']:.4f} | "
                  f"{em['time_s']:.0f}s {elapsed()}")
            self.scheduler.step(vm["val_loss"])
            if vm["val_loss"] < best_vl:
                best_vl = vm["val_loss"]
                torch.save({"model_state_dict": self.model.state_dict(), "metrics": em,
                             "config": {"target": self.target, "lambda": self.lambda_val, "loss": "cosine"}},
                           self.ckpt_dir / "model_best.pt")
            if es(vm["val_loss"]): print(f"  [STOP] Early stopping at epoch {ep}"); break
        best = min(self.history, key=lambda h: h["val_loss"])
        print(f"  ✅ Best: ep {best['epoch']} | VA: {best['val_acc']:.4f}")
        return self.history

print("✅ Trainers defined")

# %% [markdown]
# ## Cell 6: Load Base Model & Data

# %%
# Load base model with key remapping
print(f"Loading base model from checkpoint... {elapsed()}")
base_model = HybridDetector(mode="full", feature_dim=2048, dropout_rate=0.5, pretrained=False)
ckpt = torch.load(FULL_MODEL_CHECKPOINT, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
remapped = {k.replace("cbam.channel_attention.", "cbam.ca."): v for k, v in state_dict.items()}
base_model.load_state_dict(remapped)
print(f"✅ Base model loaded (with CBAM key remapping)")

# Load datasets
train_dir = DATA_ROOT / "train"
valid_dir = DATA_ROOT / "valid"
test_dir = DATA_ROOT / "test"

# Validation loader (standard, for all trainers)
val_ds = DualStreamDataset(valid_dir, split="valid", image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES_PER_CLASS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print(f"  Validation: {len(val_ds)} images, {len(val_loader)} batches")

# Naive augmentation train loader
aug_ds = AugmentedDataset(train_dir, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES_PER_CLASS)
aug_loader = DataLoader(aug_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True)
print(f"  Augmented train: {len(aug_ds)} images, {len(aug_loader)} batches")

# Consistency train loader
con_ds = ConsistencyDataset(train_dir, image_size=IMAGE_SIZE, max_samples=MAX_SAMPLES_PER_CLASS)
con_loader = DataLoader(con_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True)
print(f"  Consistency train: {len(con_ds)} images, {len(con_loader)} batches")

# Test loader (for benchmark)
test_ds = DualStreamDataset(test_dir, split="test", image_size=IMAGE_SIZE, max_samples=BENCHMARK_TEST_SAMPLES)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print(f"  Test: {len(test_ds)} images, {len(test_loader)} batches")

print(f"\n✅ All data loaded {elapsed()}")

# %% [markdown]
# ## Cell 7: Train All 7 Variants

# %%
TRAINING_RUNS = [
    {"name": "naive_aug", "type": "naive"},
    {"name": "cosine_spatial_l001", "type": "consistency", "target": "spatial", "lambda": 0.01},
    {"name": "cosine_spatial_l01",  "type": "consistency", "target": "spatial", "lambda": 0.1},
    {"name": "cosine_spatial_l10",  "type": "consistency", "target": "spatial", "lambda": 1.0},
    {"name": "cosine_dual_l001",   "type": "consistency", "target": "dual",    "lambda": 0.01},
    {"name": "cosine_dual_l01",    "type": "consistency", "target": "dual",    "lambda": 0.1},
    {"name": "cosine_dual_l10",    "type": "consistency", "target": "dual",    "lambda": 1.0},
]

training_summaries = {}

for run in TRAINING_RUNS:
    name = run["name"]
    ckpt_dir = CHECKPOINT_DIR / name
    res_dir = RESULTS_DIR / name

    print(f"\n{'#'*70}")
    print(f"#  {name.upper()}  {elapsed()}")
    print(f"{'#'*70}")

    # Skip if already trained
    if (ckpt_dir / "model_best.pt").exists():
        print(f"  [SKIP] Already trained")
        ckpt = torch.load(ckpt_dir / "model_best.pt", map_location="cpu", weights_only=False)
        training_summaries[name] = ckpt.get("metrics", {})
        continue

    # Clone base model
    model = copy.deepcopy(base_model)

    if run["type"] == "naive":
        trainer = NaiveAugTrainer(model, aug_loader, val_loader,
                                  device=DEVICE, lr=FINE_TUNE_LR,
                                  checkpoint_dir=str(ckpt_dir), results_dir=str(res_dir))
        history = trainer.fit(num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)
    else:
        trainer = ConsistencyTrainer(model, con_loader, val_loader,
                                     target=run["target"], lambda_val=run["lambda"],
                                     device=DEVICE, lr=FINE_TUNE_LR,
                                     checkpoint_dir=str(ckpt_dir), results_dir=str(res_dir))
        history = trainer.fit(num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)

    best = min(history, key=lambda h: h["val_loss"])
    training_summaries[name] = best
    del model, trainer
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"  ALL TRAINING DONE! {elapsed()}")
print(f"{'='*70}")
for name, m in training_summaries.items():
    va = m.get("val_acc", "?")
    if isinstance(va, float): va = f"{va:.4f}"
    print(f"  {name:25s} | Val Acc: {va}")

# %% [markdown]
# ## Cell 8: Benchmark Engine

# %%
class JPEGCompress:
    def __init__(self, quality=75): self.quality = quality
    def __call__(self, image): return jpeg_compress_pil(image, self.quality)

class ResizeDegradation:
    def __init__(self, scale=0.5, target_size=256):
        self.scale = scale; self.target_size = target_size
    def __call__(self, image):
        w, h = image.size
        small = (max(1, int(w*self.scale)), max(1, int(h*self.scale)))
        return image.resize(small, Image.BILINEAR).resize((self.target_size, self.target_size), Image.BILINEAR)

class CropDegradation:
    def __init__(self, crop_ratio=0.75, target_size=256):
        self.crop_ratio = crop_ratio; self.target_size = target_size
    def __call__(self, image):
        w, h = image.size
        cw, ch = int(w*self.crop_ratio), int(h*self.crop_ratio)
        l, t = (w-cw)//2, (h-ch)//2
        return image.crop((l, t, l+cw, t+ch)).resize((self.target_size, self.target_size), Image.BILINEAR)

class DegradationBenchmark:
    def __init__(self, model, test_dataset, device="cuda"):
        self.model = model.to(device); self.model.eval()
        self.dataset = test_dataset; self.device = device

    def _metrics(self, labels, preds):
        bp = [1 if p > 0.5 else 0 for p in preds]
        return {"accuracy": accuracy_score(labels, bp)*100, "auc": _safe_auc(labels, preds),
                "f1": f1_score(labels, bp, zero_division=0)}

    @torch.no_grad()
    def evaluate_clean(self, loader):
        labels, preds = [], []
        for s, f, lab in loader:
            pred = self.model(s.to(self.device), f.to(self.device))
            labels.extend(lab.tolist()); preds.extend(pred.cpu().squeeze(1).tolist())
        return self._metrics(labels, preds)

    @torch.no_grad()
    def evaluate_degraded(self, deg_fn, max_images=3000):
        labels, preds = [], []
        n = min(len(self.dataset), max_images)
        for idx in range(n):
            path, label = self.dataset.samples[idx]
            img = Image.open(path).convert("RGB").resize(
                (self.dataset.image_size, self.dataset.image_size), Image.BILINEAR)
            degraded = deg_fn(img)
            s = SPATIAL_TF(degraded).unsqueeze(0).to(self.device)
            f = DCT_TF(degraded).unsqueeze(0).to(self.device)
            pred = self.model(s, f)
            labels.append(label); preds.append(pred.cpu().squeeze().item())
        return self._metrics(labels, preds)

    def run_all(self, loader, max_images=3000):
        results = {}
        print("    ▸ Clean..."); results["Clean"] = self.evaluate_clean(loader)
        for q in JPEG_QUALITIES:
            k = f"JPEG Q={q}"; print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(JPEGCompress(q), max_images)
        for s in RESIZE_SCALES:
            k = f"Resize {int(s*100)}%"; print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(ResizeDegradation(s, IMAGE_SIZE), max_images)
        for r in CROP_RATIOS:
            k = f"Crop {int(r*100)}%"; print(f"    ▸ {k}...")
            results[k] = self.evaluate_degraded(CropDegradation(r, IMAGE_SIZE), max_images)
        print("    ▸ Combined...")
        results["Combined (Q50+Res75%)"] = self.evaluate_degraded(
            lambda img: ResizeDegradation(0.75, IMAGE_SIZE)(JPEGCompress(50)(img)), max_images)
        return results

print("✅ Benchmark engine defined")

# %% [markdown]
# ## Cell 9: Run Benchmarks

# %%
ALL_MODEL_NAMES = ["baseline"] + [r["name"] for r in TRAINING_RUNS]
all_results = {}

for name in ALL_MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"  BENCHMARKING: {name.upper()}  {elapsed()}")
    print(f"{'='*60}")

    model = HybridDetector(mode="full", feature_dim=2048, dropout_rate=0.5, pretrained=False)

    if name == "baseline":
        model.load_state_dict(remapped)  # Original baseline
        print(f"  ✅ Loaded original baseline")
    else:
        ckpt_path = CHECKPOINT_DIR / name / "model_best.pt"
        if not ckpt_path.exists():
            print(f"  ❌ No checkpoint — skipping"); del model; continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✅ Loaded {name}")

    bench = DegradationBenchmark(model, test_ds, device=DEVICE)
    results = bench.run_all(test_loader, max_images=BENCHMARK_TEST_SAMPLES*2)
    all_results[name] = results

    clean = results["Clean"]["accuracy"]
    print(f"\n  {'Condition':<28s} {'Acc%':>7s} {'Drop':>8s}")
    for cond, m in results.items():
        drop = f"({m['accuracy']-clean:+.1f})" if cond != "Clean" else ""
        print(f"  {cond:<28s} {m['accuracy']:>6.1f}% {drop:>8s}")

    del model, bench; torch.cuda.empty_cache()

with open(RESULTS_DIR / "all_benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ All benchmarks done! {elapsed()}")

# %% [markdown]
# ## Cell 10: Results Tables

# %%
models_present = [m for m in ALL_MODEL_NAMES if m in all_results]
all_conditions = []
for mr in all_results.values():
    for c in mr:
        if c not in all_conditions: all_conditions.append(c)

print("\n" + "=" * 120)
print("  CONSISTENCY TRAINING (COSINE) — COMPLETE RESULTS")
print("=" * 120)

header = f"{'Condition':<28s}"
for m in models_present: header += f" | {m:>16s}"
print(header); print("-" * len(header))

for cond in all_conditions:
    row = f"{cond:<28s}"
    for m in models_present:
        row += f" | {all_results[m].get(cond, {}).get('accuracy', 0):>15.1f}%"
    print(row)

# Drop table
print(f"\n  ACCURACY DROP FROM CLEAN (pp):")
for cond in all_conditions:
    if cond == "Clean": continue
    row = f"{cond:<28s}"
    for m in models_present:
        clean = all_results[m]["Clean"]["accuracy"]
        drop = all_results[m].get(cond, {}).get("accuracy", clean) - clean
        row += f" | {drop:>+15.1f}%"
    print(row)

# Best lambda summary
print(f"\n  BEST λ UNDER JPEG Q=50:")
for name in models_present:
    if "cosine_" in name and "JPEG Q=50" in all_results[name]:
        acc = all_results[name]["JPEG Q=50"]["accuracy"]
        print(f"    {name:25s}: {acc:.1f}%")

# %% [markdown]
# ## Cell 11: Comparison Plots

# %%
# ────── Plot 1: JPEG Q vs Accuracy — All Models ──────
jpeg_conds = ["Clean"] + [f"JPEG Q={q}" for q in JPEG_QUALITIES]
jpeg_labels = ["Clean", "Q=100", "Q=75", "Q=50", "Q=25"]

fig, ax = plt.subplots(figsize=(14, 8))
cmap = plt.cm.tab10
for i, name in enumerate(models_present):
    accs = [all_results[name].get(c, {}).get("accuracy", 0) for c in jpeg_conds]
    style = "-o" if name == "baseline" else "--s" if name == "naive_aug" else "-^"
    lw = 3 if name in ("baseline", "naive_aug") else 1.5
    ax.plot(range(len(jpeg_labels)), accs, style, color=cmap(i), label=name, linewidth=lw, markersize=7)
ax.set_xticks(range(len(jpeg_labels))); ax.set_xticklabels(jpeg_labels)
ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("JPEG Quality")
ax.set_title("Consistency Training (Cosine): JPEG Compression Robustness", fontsize=14, fontweight="bold")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cosine_jpeg_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); print(f"✅ Saved cosine_jpeg_comparison.png")

# ────── Plot 2: Bar chart — JPEG Q=50 accuracy for all models ──────
fig, ax = plt.subplots(figsize=(14, 6))
names = [m for m in models_present if "JPEG Q=50" in all_results[m]]
accs = [all_results[m]["JPEG Q=50"]["accuracy"] for m in names]
colors = ["#E53935" if m == "baseline" else "#FF9800" if m == "naive_aug" else
          "#4CAF50" if "spatial" in m else "#2196F3" for m in names]
bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="black", linewidth=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim([0, 110])
ax.set_title("JPEG Q=50 Accuracy: Baseline vs Consistency Training (Cosine)", fontsize=13, fontweight="bold")
ax.axhline(y=accs[0], color="#E53935", linestyle="--", alpha=0.5, label=f"Baseline: {accs[0]:.1f}%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cosine_jpeg50_bar.png", dpi=150, bbox_inches="tight")
plt.show(); print(f"✅ Saved cosine_jpeg50_bar.png")

# ────── Plot 3: Lambda sweep — spatial vs dual ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Cosine Consistency: λ Sweep Under JPEG Q=50", fontsize=14, fontweight="bold")

for ax_idx, target in enumerate(["spatial", "dual"]):
    ax = axes[ax_idx]
    lvals, accs_q50, accs_q25 = [], [], []
    for lv in LAMBDA_VALUES:
        if lv == 0.01: name = f"cosine_{target}_l001"
        elif lv == 0.1: name = f"cosine_{target}_l01"
        elif lv == 1.0: name = f"cosine_{target}_l10"
        if name in all_results:
            lvals.append(str(lv))
            accs_q50.append(all_results[name].get("JPEG Q=50", {}).get("accuracy", 0))
            accs_q25.append(all_results[name].get("JPEG Q=25", {}).get("accuracy", 0))
    if lvals:
        x = range(len(lvals))
        ax.bar([xi-0.15 for xi in x], accs_q50, 0.3, label="JPEG Q=50", color="#FF9800")
        ax.bar([xi+0.15 for xi in x], accs_q25, 0.3, label="JPEG Q=25", color="#F44336")
        if "baseline" in all_results:
            bl50 = all_results["baseline"].get("JPEG Q=50", {}).get("accuracy", 0)
            ax.axhline(y=bl50, color="gray", linestyle="--", label=f"Baseline Q=50: {bl50:.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(lvals)
        ax.set_xlabel("λ"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{target.title()} Branch Consistency")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cosine_lambda_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); print(f"✅ Saved cosine_lambda_sweep.png")

# ────── Plot 4: Heatmap — Accuracy drop improvement over baseline ──────
conds_jpeg = [f"JPEG Q={q}" for q in [75, 50, 25]] + ["Combined (Q50+Res75%)"]
consistency_models = [m for m in models_present if m not in ("baseline",)]

if "baseline" in all_results and consistency_models:
    improvement = []
    for m in consistency_models:
        row = []
        for c in conds_jpeg:
            bl_acc = all_results["baseline"].get(c, {}).get("accuracy", 0)
            m_acc = all_results[m].get(c, {}).get("accuracy", 0)
            row.append(m_acc - bl_acc)
        improvement.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(np.array(improvement), annot=True, fmt="+.1f", cmap="RdYlGn", center=0,
                xticklabels=conds_jpeg, yticklabels=consistency_models,
                ax=ax, linewidths=0.5, annot_kws={"fontsize": 9})
    ax.set_title("Accuracy Improvement Over Baseline (pp) — Positive = Better", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cosine_improvement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show(); print(f"✅ Saved cosine_improvement_heatmap.png")

# %% [markdown]
# ## Cell 12: Key Findings

# %%
print(f"\n{'='*70}")
print(f"  🛠️ CONSISTENCY TRAINING (COSINE) — KEY FINDINGS  {elapsed()}")
print(f"{'='*70}")

if "baseline" in all_results:
    bl_clean = all_results["baseline"]["Clean"]["accuracy"]
    bl_q50 = all_results["baseline"].get("JPEG Q=50", {}).get("accuracy", 0)
    bl_q25 = all_results["baseline"].get("JPEG Q=25", {}).get("accuracy", 0)
    print(f"\n📊 Baseline: Clean={bl_clean:.1f}% | Q=50={bl_q50:.1f}% | Q=25={bl_q25:.1f}%")

    # Find best model for JPEG Q=50
    best_name, best_acc = "baseline", bl_q50
    for m in models_present:
        acc = all_results[m].get("JPEG Q=50", {}).get("accuracy", 0)
        if acc > best_acc:
            best_name, best_acc = m, acc

    print(f"\n🏆 Best model under JPEG Q=50: {best_name} ({best_acc:.1f}%)")
    print(f"   Improvement over baseline: {best_acc - bl_q50:+.1f}pp")

    if "naive_aug" in all_results:
        na_q50 = all_results["naive_aug"].get("JPEG Q=50", {}).get("accuracy", 0)
        print(f"\n📊 Naive augmentation Q=50: {na_q50:.1f}% ({na_q50 - bl_q50:+.1f}pp vs baseline)")
        print(f"   Best consistency Q=50:   {best_acc:.1f}% ({best_acc - na_q50:+.1f}pp vs naive aug)")

    # Clean accuracy check (did we sacrifice clean performance?)
    print(f"\n📊 Clean accuracy preservation:")
    for m in models_present:
        clean = all_results[m]["Clean"]["accuracy"]
        drop = clean - bl_clean
        status = "✅" if abs(drop) < 2 else "⚠️"
        print(f"    {status} {m:25s}: {clean:.1f}% ({drop:+.1f}pp)")

# Save final
final = {
    "experiment": "consistency_training_cosine",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {"lr": FINE_TUNE_LR, "max_samples": MAX_SAMPLES_PER_CLASS,
               "lambdas": LAMBDA_VALUES, "epochs": NUM_EPOCHS},
    "results": all_results,
}
with open(RESULTS_DIR / "consistency_cosine_complete.json", "w") as f:
    json.dump(final, f, indent=2)

print(f"\n{'='*70}")
print(f"  ✅ COSINE CONSISTENCY TRAINING COMPLETE! {elapsed()}")
print(f"  Compare MSE vs Cosine results for your paper!")
print(f"{'='*70}")
