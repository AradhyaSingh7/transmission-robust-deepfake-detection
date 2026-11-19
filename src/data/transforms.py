"""
Forensic Transformations: 2D DCT Log-Spectrum and Lossy Degradations
"""
import io
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy.fft import dctn


class DCTTransform:
    """Computes the 2D Discrete Cosine Transform (DCT) of an image.
    
    Transforms the spatial image into frequency representations.
    Because raw DCT coefficients span orders of magnitude, we apply:
        1. Log-scale compression: C_log = sign(C) * log(1 + |C|)
        2. Channel min-max normalization into [0, 1] range.
    """
    def __init__(self, channels='rgb'):
        self.channels = channels

    def __call__(self, image):
        """
        Args:
            image: PIL Image (RGB) or numpy array
        Returns:
            torch.Tensor of shape (3, H, W) or (1, H, W)
        """
        if isinstance(image, Image.Image):
            arr = np.array(image, dtype=np.float32)
        elif isinstance(image, torch.Tensor):
            arr = image.permute(1, 2, 0).numpy() * 255.0
        else:
            arr = image.astype(np.float32)

        if self.channels == 'gray':
            if len(arr.shape) == 3:
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            else:
                gray = arr
            dct_coeffs = dctn(gray, type=2, norm='ortho')
            dct_log = np.sign(dct_coeffs) * np.log1p(np.abs(dct_coeffs))
            denom = dct_log.max() - dct_log.min() + 1e-8
            dct_norm = (dct_log - dct_log.min()) / denom
            return torch.from_numpy(dct_norm).unsqueeze(0).float()
        else:
            # Per-channel DCT (RGB)
            dct_chans = []
            for c in range(3):
                chan = arr[:, :, c] if len(arr.shape) == 3 else arr
                dct_coeffs = dctn(chan, type=2, norm='ortho')
                dct_log = np.sign(dct_coeffs) * np.log1p(np.abs(dct_coeffs))
                denom = dct_log.max() - dct_log.min() + 1e-8
                dct_norm = (dct_log - dct_log.min()) / denom
                dct_chans.append(dct_norm)
            return torch.from_numpy(np.stack(dct_chans, axis=0)).float()


class JPEGCompress:
    """Simulates social media transmission via in-memory lossy JPEG compression."""
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL Image")
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class WebPCompress:
    """Simulates modern web media transmission via in-memory lossy WebP compression."""
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL Image")
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class ResizeCompress:
    """Simulates platform thumbnail downsampling and resizing."""
    def __init__(self, scale=0.75):
        self.scale = scale

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL Image")
        orig_w, orig_h = img.size
        new_w = max(int(orig_w * self.scale), 16)
        new_h = max(int(orig_h * self.scale), 16)
        downsampled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return downsampled.resize((orig_w, orig_h), Image.Resampling.BILINEAR)


class CompoundDegradation:
    """Compound stress: simultaneous JPEG compression and geometric downsampling."""
    def __init__(self, jpeg_quality=50, resize_scale=0.75):
        self.jpeg = JPEGCompress(quality=jpeg_quality)
        self.resize = ResizeCompress(scale=resize_scale)

    def __call__(self, img):
        return self.resize(self.jpeg(img))


def get_spatial_transform(image_size=256):
    """Standard spatial normalization pipeline for ResNet-50."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
