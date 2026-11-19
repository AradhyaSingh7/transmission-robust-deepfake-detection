#!/usr/bin/env python
"""
================================================================================
DEMO: Dual-Stream Spatial-Frequency Deepfake Detection with Dynamic Gated Fusion
================================================================================
Usage:
    python demo.py --image samples/fake/fake_01.jpg
    python demo.py --image samples/real/real_01.jpg --degrade jpeg50
    python demo.py --sample-all
================================================================================
"""
import argparse
import sys
import os
import tempfile
from pathlib import Path

# Set Matplotlib config dir to avoid non-writable warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))

from PIL import Image
import numpy as np

# Matplotlib headless setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from src.models.dynamic_gate import GatedHybridDetector
    from src.data.transforms import (
        DCTTransform,
        JPEGCompress,
        WebPCompress,
        ResizeCompress,
        CompoundDegradation,
        get_spatial_transform,
    )
    from src.config import DEVICE, IMAGE_SIZE
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def apply_degradation(img, degradation_name):
    """Applies specified transmission degradation to PIL image."""
    deg = degradation_name.lower()
    if deg in ("none", "clean"):
        return img, "None (Clean)"
    elif deg == "jpeg75":
        return JPEGCompress(quality=75)(img), "JPEG (Q=75)"
    elif deg == "jpeg50":
        return JPEGCompress(quality=50)(img), "JPEG (Q=50)"
    elif deg == "jpeg25":
        return JPEGCompress(quality=25)(img), "JPEG (Q=25)"
    elif deg == "webp" or deg == "webp25":
        return WebPCompress(quality=25)(img), "WebP (Q=25)"
    elif deg == "resize50":
        return ResizeCompress(scale=0.50)(img), "Downsampled 50%"
    elif deg == "resize75":
        return ResizeCompress(scale=0.75)(img), "Downsampled 75%"
    elif deg == "compound":
        return CompoundDegradation(jpeg_quality=50, resize_scale=0.75)(img), "Compound (JPEG Q=50 + Resize 75%)"
    else:
        return img, f"Unknown ({degradation_name})"


def run_inference(image_path, degradation="none", output_plot="demo_output.png"):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    print("=" * 80)
    print(" DUAL-STREAM SPATIAL-FREQUENCY DEEPFAKE DETECTOR (DYNAMIC GATED FUSION)")
    print("=" * 80)
    print(f" Target Image:      {image_path}")

    # Load and preprocess
    orig_img = Image.open(image_path).convert('RGB')
    orig_img = orig_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    degraded_img, deg_label = apply_degradation(orig_img, degradation)
    print(f" Channel Stress:    {deg_label}")

    # Compute DCT representation
    dct_fn = DCTTransform(channels='rgb')
    freq_tensor = dct_fn(degraded_img)

    spatial_fn = get_spatial_transform(IMAGE_SIZE)
    spatial_tensor = spatial_fn(degraded_img)

    # Convert DCT for visualization (grayscale projection)
    dct_vis = freq_tensor.mean(dim=0).numpy()

    # Model inference
    model = GatedHybridDetector(pretrained=False)
    model.eval()

    # Search for available checkpoints
    ckpt_candidates = [
        Path("weights/gated_model_best.pt"),
        Path("../Deepfake/checkpoints/full/model_best"),
        Path("c:/CV-Project/Deepfake/checkpoints/full/model_best"),
    ]
    ckpt_loaded = False
    for ckpt_path in ckpt_candidates:
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
                model.load_state_dict(state_dict, strict=False)
                print(f" Checkpoint:        Loaded from {ckpt_path}")
                ckpt_loaded = True
                break
            except Exception as e:
                pass

    if not ckpt_loaded:
        print(" Checkpoint:        Using initialized architecture weights (Demonstration mode)")

    with torch.no_grad():
        s_in = spatial_tensor.unsqueeze(0)
        f_in = freq_tensor.unsqueeze(0)
        prob, alpha = model(s_in, f_in)
        fake_prob = float(prob.item())
        alpha_val = float(alpha.item())

    # Calibrate / determine verdict
    is_fake = fake_prob >= 0.5
    confidence = fake_prob if is_fake else (1.0 - fake_prob)
    verdict = "FAKE (Synthetic Face Detected)" if is_fake else "REAL (Authentic Facial Identity)"

    # Print summary
    print("-" * 80)
    print(f" Classification:    {verdict}")
    print(f" Confidence:        {confidence * 100:.2f}% (Raw Probability: {fake_prob:.4f})")
    print(f" Gating Alpha (α):  {alpha_val:.3f} [Spatial: {alpha_val*100:.1f}% | Frequency: {(1.0-alpha_val)*100:.1f}%]")

    # Forensic diagnostic summary
    if alpha_val > 0.65:
        routing_reason = "Dynamic gate allocated higher weight to Spatial stream (Frequency suppressed due to lossy artifacts)."
    elif alpha_val < 0.35:
        routing_reason = "Dynamic gate allocated higher weight to Frequency stream (Spatial cues degraded by compression/blur)."
    else:
        routing_reason = "Balanced multi-stream synergy: both spatial and frequency domains contribute symmetrically."
    print(f" Routing Diagnosis: {routing_reason}")

    # Generate professional visual inspection plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # 1. Spatial image
    axes[0].imshow(degraded_img)
    axes[0].set_title(f"Input RGB ({deg_label})\nVerdict: {verdict.split()[0]} ({confidence*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. 2D DCT log-spectrum
    im = axes[1].imshow(dct_vis, cmap="inferno")
    axes[1].set_title("2D DCT Log-Magnitude Spectrum\n(Orthogonal Frequency Domain)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Dynamic Gating & Confidence
    bars = axes[2].barh(["Spatial (α)", "Frequency (1-α)"], [alpha_val, 1.0 - alpha_val], color=["#2b5c8f", "#c0392b"], height=0.45)
    axes[2].set_xlim(0, 1.0)
    axes[2].set_xlabel("Dynamic Gating Allocation", fontsize=10, fontweight="bold")
    axes[2].set_title(f"Dynamic Gated Routing\nα = {alpha_val:.3f}", fontsize=11, fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        axes[2].text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close()

    print("-" * 80)
    print(f" Forensic Plot:     Saved to '{output_plot}'")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Dual-Stream Deepfake Detector CLI")
    parser.add_argument("--image", type=str, default="samples/fake/fake_01.jpg", help="Path to facial image")
    parser.add_argument("--degrade", type=str, default="none",
                        choices=["none", "clean", "jpeg75", "jpeg50", "jpeg25", "webp", "resize50", "resize75", "compound"],
                        help="Optional transmission degradation stress to apply")
    parser.add_argument("--output", type=str, default="demo_output.png", help="Path to save forensic plot")
    parser.add_argument("--sample-all", action="store_true", help="Run inference sequentially over all sample images")

    args = parser.parse_args()

    if args.sample_all:
        sample_dir = Path("samples")
        images = list(sample_dir.glob("*/*.jpg"))
        print(f"Running automated audit on {len(images)} sample images...\n")
        for idx, img_p in enumerate(images, 1):
            out_p = f"demo_output_sample_{idx}.png"
            run_inference(img_p, degradation=args.degrade, output_plot=out_p)
            print()
    else:
        run_inference(args.image, degradation=args.degrade, output_plot=args.output)


if __name__ == "__main__":
    main()
