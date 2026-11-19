"""
Dual-stream dataset for the hybrid spatial-frequency detector.

Returns BOTH a spatial (ImageNet-normalized RGB) tensor and a frequency
(log-scaled DCT) tensor from every image, so the model receives matched
pairs through a single DataLoader.

Augmentation (random horizontal flip) is applied to the raw PIL image
*before* the split into spatial/frequency paths, ensuring the two
branches always see the same geometric view of the image.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .transforms import DCTTransform


class DualStreamDataset(Dataset):
    """Dataset that yields (spatial_tensor, frequency_tensor, label) triples.

    Parameters
    ----------
    root_dir : str or Path
        Path to a split directory containing ``real/`` and ``fake/`` subdirs.
    split : str
        One of ``'train'``, ``'valid'``, ``'test'``.  Controls whether
        random augmentation is applied.
    image_size : int
        Target spatial resolution (images are resized to this square size).
    max_samples : int or None
        If set, limits the number of images loaded **per class** (useful
        for quick smoke-tests).
    """

    def __init__(self, root_dir, split="train", image_size=256, max_samples=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        # --- Collect image paths and labels ---
        self.samples: list[tuple[Path, int]] = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Directory not found: {class_dir}")

            image_files = sorted(
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            )
            if max_samples is not None:
                image_files = image_files[:max_samples]

            for img_path in image_files:
                self.samples.append((img_path, label_idx))

        # --- Spatial branch transform (ImageNet normalization) ---
        self.spatial_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.dct_transform = DCTTransform(channels="rgb")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # --- Training-time augmentation (applied BEFORE branch split) ---
        if self.split == "train":
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # --- Spatial branch ---
        spatial = self.spatial_transform(image)  # (3, H, W)

        # --- Frequency branch ---
        frequency = self.dct_transform(image)  # (3, H, W)

        return spatial, frequency, torch.tensor(label, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_class_counts(self):
        """Return a dict with counts of ``real``, ``fake``, and ``total``."""
        labels = [s[1] for s in self.samples]
        return {
            "real": labels.count(0),
            "fake": labels.count(1),
            "total": len(labels),
        }


# ======================================================================
# Convenience: build DataLoaders for train / valid / test
# ======================================================================


def get_dataloaders(
    data_root,
    image_size=256,
    batch_size=32,
    num_workers=2,
    max_samples=None,
):
    """Create DataLoaders for all three splits.

    Parameters
    ----------
    data_root : str or Path
        Root that contains ``train/``, ``valid/``, ``test/`` subdirectories.
    image_size : int
        Target spatial resolution.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    max_samples : int or None
        Per-class cap (for debugging).

    Returns
    -------
    dict
        ``{'train': DataLoader, 'valid': DataLoader, 'test': DataLoader}``
    """
    loaders = {}
    for split in ("train", "valid", "test"):
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            print(f"  [!] Split directory not found, skipping: {split_dir}")
            continue

        ds = DualStreamDataset(
            root_dir=split_dir,
            split=split,
            image_size=image_size,
            max_samples=max_samples,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        counts = ds.get_class_counts()
        print(
            f"  {split:5s} -> {counts['total']:,} images "
            f"(real: {counts['real']:,}, fake: {counts['fake']:,})  "
            f"| {len(loaders[split])} batches"
        )

    return loaders
