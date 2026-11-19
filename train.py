#!/usr/bin/env python
"""
================================================================================
TRAINING: Dual-Stream Deepfake Detector Training CLI
================================================================================
Usage:
    python train.py --mode full --epochs 10 --batch-size 32
    python train.py --smoke-test
    python train.py --mode gated --epochs 8
================================================================================
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from src.config import DEVICE, IMAGE_SIZE, FEATURE_DIM, DROPOUT_RATE
from src.models.detector import HybridDetector
from src.models.dynamic_gate import GatedHybridDetector
from src.data.dataset import FaceDataset
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, criterion, optimizer, is_gated=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for s_in, f_in, labels in dataloader:
        s_in = s_in.to(DEVICE)
        f_in = f_in.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        if is_gated:
            preds, alpha = model(s_in, f_in)
        else:
            preds = model(s_in, f_in)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * s_in.size(0)
        predicted = (preds >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Spatial-Frequency Deepfake Detector")
    parser.add_argument("--mode", type=str, default="gated",
                        choices=["gated", "full", "spatial_only", "frequency_only", "fusion_no_attention"],
                        help="Model architecture to train (default: gated)")
    parser.add_argument("--data-root", type=str, default="samples",
                        help="Path to dataset containing real/ and fake/ subfolders")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick 1-epoch verification on local sample data")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 2
        args.data_root = "samples"

    print("=" * 80)
    print(" DUAL-STREAM DEEPFAKE DETECTOR: TRAINING PIPELINE")
    print("=" * 80)
    print(f" Architecture:     {args.mode}")
    print(f" Training Data:    {args.data_root}")
    print(f" Device:           {DEVICE}")
    print(f" Epochs:           {args.epochs}")
    print(f" Batch Size:       {args.batch_size}")
    print(f" Learning Rate:    {args.lr}")
    print("-" * 80)

    # Initialize dataset
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"Error: Data path '{data_path}' does not exist.")
        sys.exit(1)

    dataset = FaceDataset(root_dir=data_path, image_size=IMAGE_SIZE)
    if len(dataset) == 0:
        print(f"Error: No images found in {data_path}/real or {data_path}/fake")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f" Loaded {len(dataset)} training samples.")

    # Initialize model
    is_gated = (args.mode == "gated")
    if is_gated:
        model = GatedHybridDetector(feature_dim=FEATURE_DIM, dropout_rate=DROPOUT_RATE, pretrained=False)
    else:
        model = HybridDetector(mode=args.mode, feature_dim=FEATURE_DIM, dropout_rate=DROPOUT_RATE, pretrained=False)

    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, loader, criterion, optimizer, is_gated=is_gated)
        print(f" Epoch [{epoch}/{args.epochs}] -> Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    # Save checkpoint
    ckpt_path = out_dir / f"{args.mode}_model_latest.pt"
    torch.save({
        "epoch": args.epochs,
        "mode": args.mode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)

    print("-" * 80)
    print(f" Training complete! Checkpoint saved to: {ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
