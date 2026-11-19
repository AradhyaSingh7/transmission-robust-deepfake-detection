"""
Training entry point — launch baseline training from the command line.

Usage examples::

    # Full hybrid model (default)
    python -m training.train

    # Spatial-only ablation
    python -m training.train --mode spatial_only

    # Quick smoke test (100 images per class, 3 epochs)
    python -m training.train --max_samples 100 --epochs 3

    # Custom data root (e.g. on Colab)
    python -m training.train --data_root /content/data/real-vs-fake

All hyperparameters fall back to ``config.py`` defaults when not
specified on the command line.
"""

import argparse
import sys
import os

# Ensure the ``src/`` package is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config
from data.dual_dataset import get_dataloaders
from models.detector import HybridDetector
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the hybrid deepfake detector (baseline)."
    )

    # --- Model ---
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "spatial_only", "frequency_only", "fusion_no_attention"],
        help="Architecture variant to train (default: full)",
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights (for debugging)",
    )

    # --- Data ---
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root directory (default: from config.py)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit images per class (for quick smoke tests)",
    )

    # --- Training ---
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    p.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")

    # --- Output ---
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results directory",
    )

    # --- Device ---
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu (default: auto-detect)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # --- Resolve settings (CLI > config.py defaults) ---
    data_root = args.data_root or str(config.DATA_ROOT)
    epochs = args.epochs or config.NUM_EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE
    lr = args.lr or config.LEARNING_RATE
    patience = args.patience or config.EARLY_STOPPING_PATIENCE
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    device = args.device or config.DEVICE
    checkpoint_dir = args.checkpoint_dir or str(config.CHECKPOINT_DIR)
    results_dir = args.results_dir or str(config.RESULTS_DIR)
    pretrained = not args.no_pretrained

    # Tag output dirs with mode name to avoid overwriting between ablations
    mode = args.mode
    checkpoint_dir = os.path.join(checkpoint_dir, mode)
    results_dir = os.path.join(results_dir, mode)

    # --- Banner ---
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION -- BASELINE TRAINING")
    print("=" * 60)
    print(f"  Mode:          {mode}")
    print(f"  Data root:     {data_root}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Patience:      {patience}")
    print(f"  Device:        {device}")
    print(f"  Pretrained:    {pretrained}")
    print(f"  Checkpoints:   {checkpoint_dir}")
    print(f"  Results:       {results_dir}")
    if args.max_samples:
        print(f"  Max samples:   {args.max_samples} per class (SMOKE TEST)")
    print("=" * 60)

    # --- Data ---
    print("\n[*] Loading datasets...")
    loaders = get_dataloaders(
        data_root=data_root,
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=args.max_samples,
    )

    if "train" not in loaders or "valid" not in loaders:
        print("[X] Could not load train and/or valid splits. Aborting.")
        sys.exit(1)

    # --- Model ---
    print("\n[*] Building model...")
    model = HybridDetector(
        mode=mode,
        feature_dim=config.FEATURE_DIM,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=pretrained,
    )
    params = model.get_num_parameters()
    print(f"  {model}")
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["valid"],
        device=device,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    # --- Train ---
    history = trainer.fit(num_epochs=epochs, patience=patience)

    print("[OK] Done. Check results at:", results_dir)


if __name__ == "__main__":
    main()
