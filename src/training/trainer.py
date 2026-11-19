"""
Training engine for the Hybrid Deepfake Detector.

Provides:
- ``Trainer`` -- full training loop with validation, early stopping,
  checkpoint saving, metric tracking, and training-curve plots.
- ``EarlyStopping`` -- patience-based early termination on validation loss.

Usage (standalone)::

    python -m training.trainer          # see __main__ block at bottom

Or import into a Colab notebook::

    from training.trainer import Trainer
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# ======================================================================
# Early stopping
# ======================================================================


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``'min'`` for metrics like loss, ``'max'`` for AUC / accuracy.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ======================================================================
# Trainer
# ======================================================================


class Trainer:
    """Complete training pipeline for :class:`HybridDetector`.

    Parameters
    ----------
    model : nn.Module
        A ``HybridDetector`` instance (any mode).
    train_loader, val_loader : DataLoader
        Must yield ``(spatial, frequency, label)`` triples
        (use :class:`DualStreamDataset`).
    device : str
        ``'cuda'`` or ``'cpu'``.
    lr : float
        Initial learning rate for Adam.
    checkpoint_dir : Path or str
        Where to save model checkpoints.
    results_dir : Path or str
        Where to save CSV logs and plots.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        *,
        device: str = "cuda",
        lr: float = 1e-4,
        checkpoint_dir: str | Path = "checkpoints",
        results_dir: str | Path = "results",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Metric history -- one dict per epoch
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Training loop for one epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()

        running_loss = 0.0
        all_labels: list[float] = []
        all_preds: list[float] = []
        num_batches = 0

        for batch_idx, (spatial, frequency, labels) in enumerate(self.train_loader):
            spatial = spatial.to(self.device)
            frequency = frequency.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # (B, 1)

            # Forward
            predictions = self.model(spatial, frequency)
            loss = self.criterion(predictions, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Collect for metrics
            all_labels.extend(labels.detach().cpu().squeeze(1).tolist())
            all_preds.extend(predictions.detach().cpu().squeeze(1).tolist())

            # Progress print every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg = running_loss / num_batches
                print(
                    f"    Epoch {epoch} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg:.4f}"
                )

        # Epoch-level metrics
        avg_loss = running_loss / max(num_batches, 1)
        binary_preds = [1.0 if p > 0.5 else 0.0 for p in all_preds]
        metrics = {
            "train_loss": avg_loss,
            "train_acc": accuracy_score(all_labels, binary_preds),
            "train_auc": _safe_auc(all_labels, all_preds),
            "train_f1": f1_score(all_labels, binary_preds, zero_division=0),
        }
        return metrics

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()

        running_loss = 0.0
        all_labels: list[float] = []
        all_preds: list[float] = []
        num_batches = 0

        for spatial, frequency, labels in self.val_loader:
            spatial = spatial.to(self.device)
            frequency = frequency.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            predictions = self.model(spatial, frequency)
            loss = self.criterion(predictions, labels)

            running_loss += loss.item()
            num_batches += 1

            all_labels.extend(labels.cpu().squeeze(1).tolist())
            all_preds.extend(predictions.cpu().squeeze(1).tolist())

        avg_loss = running_loss / max(num_batches, 1)
        binary_preds = [1.0 if p > 0.5 else 0.0 for p in all_preds]

        metrics = {
            "val_loss": avg_loss,
            "val_acc": accuracy_score(all_labels, binary_preds),
            "val_auc": _safe_auc(all_labels, all_preds),
            "val_f1": f1_score(all_labels, binary_preds, zero_division=0),
            "val_precision": precision_score(all_labels, binary_preds, zero_division=0),
            "val_recall": recall_score(all_labels, binary_preds, zero_division=0),
        }
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "best"):
        """Save model weights, optimizer state, and metadata."""
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "model_mode": getattr(self.model, "mode", "unknown"),
            },
            path,
        )
        print(f"    [SAVE] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path):
        """Restore model and optimizer from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(
            f"    [OK] Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(val_acc={ckpt['metrics'].get('val_acc', '?'):.4f})"
        )
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _save_history_csv(self):
        """Write the full metric history to a CSV file."""
        if not self.history:
            return
        path = self.results_dir / "training_history.csv"
        keys = self.history[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"    [CSV] History saved: {path}")

    def _save_training_plots(self):
        """Generate and save loss / accuracy / AUC curves."""
        if len(self.history) < 2:
            return

        epochs = [h["epoch"] for h in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

        # --- Loss ---
        axes[0].plot(epochs, [h["train_loss"] for h in self.history], "b-o", label="Train")
        axes[0].plot(epochs, [h["val_loss"] for h in self.history], "r-o", label="Valid")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Accuracy ---
        axes[1].plot(epochs, [h["train_acc"] for h in self.history], "b-o", label="Train")
        axes[1].plot(epochs, [h["val_acc"] for h in self.history], "r-o", label="Valid")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0.5, 1.0])

        # --- AUC ---
        axes[2].plot(epochs, [h["train_auc"] for h in self.history], "b-o", label="Train")
        axes[2].plot(epochs, [h["val_auc"] for h in self.history], "r-o", label="Valid")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC-ROC")
        axes[2].set_title("AUC-ROC")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0.5, 1.0])

        plt.tight_layout()
        path = self.results_dir / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [PLOT] Training curves saved: {path}")

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        num_epochs: int = 20,
        patience: int = 5,
        save_every: int = 5,
    ) -> list[dict]:
        """Run the full training loop.

        Parameters
        ----------
        num_epochs : int
            Maximum number of epochs.
        patience : int
            Early-stopping patience (epochs without val_loss improvement).
        save_every : int
            Save a periodic checkpoint every N epochs (in addition to
            the best-model checkpoint).

        Returns
        -------
        list[dict]
            The full metric history.
        """
        early_stopping = EarlyStopping(patience=patience, mode="min")
        best_val_loss = float("inf")

        print(f"\n{'=' * 60}")
        print(f"  Starting training: {num_epochs} epochs, patience={patience}")
        print(f"  Model mode: {getattr(self.model, 'mode', '?')}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Valid batches: {len(self.val_loader)}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = self._train_one_epoch(epoch)

            # --- Validate ---
            val_metrics = self._validate()

            # --- Merge metrics ---
            epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            epoch_metrics["time_s"] = time.time() - epoch_start
            self.history.append(epoch_metrics)

            # --- Console output ---
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f}  Acc: {train_metrics['train_acc']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}  Acc: {val_metrics['val_acc']:.4f}  "
                f"AUC: {val_metrics['val_auc']:.4f}  F1: {val_metrics['val_f1']:.4f} | "
                f"{epoch_metrics['time_s']:.1f}s"
            )

            # --- Learning-rate scheduling ---
            self.scheduler.step(val_metrics["val_loss"])

            # --- Best-model checkpoint ---
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, epoch_metrics, tag="best")

            # --- Periodic checkpoint ---
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, epoch_metrics, tag=f"epoch_{epoch}")

            # --- Early stopping ---
            if early_stopping(val_metrics["val_loss"]):
                print(f"\n  [STOP] Early stopping at epoch {epoch} (patience={patience})")
                break

        # --- Final saves ---
        self.save_checkpoint(epoch, epoch_metrics, tag="last")
        self._save_history_csv()
        self._save_training_plots()

        # --- Final summary ---
        best = min(self.history, key=lambda h: h["val_loss"])
        print(f"\n{'=' * 60}")
        print(f"  Training complete!")
        print(
            f"  Best epoch: {best['epoch']} | "
            f"Val Loss: {best['val_loss']:.4f} | "
            f"Val Acc: {best['val_acc']:.4f} | "
            f"Val AUC: {best['val_auc']:.4f}"
        )
        print(f"{'=' * 60}\n")

        # Save summary JSON
        summary = {
            "total_epochs": epoch,
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "best_val_auc": best["val_auc"],
            "best_val_f1": best["val_f1"],
            "model_mode": getattr(self.model, "mode", "unknown"),
        }
        with open(self.results_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return self.history


# ======================================================================
# Helpers
# ======================================================================


def _safe_auc(labels, preds) -> float:
    """Compute AUC-ROC, returning 0.0 if only one class is present."""
    try:
        return roc_auc_score(labels, preds)
    except ValueError:
        return 0.0
