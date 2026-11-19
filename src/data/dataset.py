"""
PyTorch Dataset Loaders for Dual-Stream Architecture
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import DCTTransform, get_spatial_transform


class FaceDataset(Dataset):
    """Dataset yielding (spatial_tensor, frequency_tensor, label)."""
    def __init__(self, root_dir, image_size=256, degradation=None, max_samples=None):
        """
        Args:
            root_dir: Directory containing 'real' and 'fake' subfolders
            image_size: Target square image dimension
            degradation: Optional callable degradation transform (e.g. JPEGCompress)
            max_samples: Optional limit on total samples per class
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.degradation = degradation
        self.spatial_transform = get_spatial_transform(image_size)
        self.dct_transform = DCTTransform(channels='rgb')

        self.samples = []
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        if real_dir.exists():
            real_files = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
            if max_samples:
                real_files = real_files[:max_samples]
            for p in real_files:
                self.samples.append((p, 0.0))  # 0.0 = Real

        if fake_dir.exists():
            fake_files = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
            if max_samples:
                fake_files = fake_files[:max_samples]
            for p in fake_files:
                self.samples.append((p, 1.0))  # 1.0 = Fake

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        if self.degradation is not None:
            img = self.degradation(img)

        # Spatial stream tensor
        spatial_tensor = self.spatial_transform(img)

        # Frequency stream tensor (DCT)
        freq_tensor = self.dct_transform(img)

        return spatial_tensor, freq_tensor, torch.tensor([label], dtype=torch.float32)
