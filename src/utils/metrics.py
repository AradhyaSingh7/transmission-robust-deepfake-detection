"""
Metrics: Classification Accuracy, ROC-AUC, True Negative/Positive Rates, Score Separation
"""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """Computes comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake)
        y_pred_prob: Predicted probabilities of synthetic manipulation
        threshold: Decision boundary (default 0.5)
    Returns:
        dict containing accuracy, auc, f1, tnr, tpr, and mean scores
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5  # Only one class present

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tnr = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    tpr = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

    real_scores = y_pred_prob[y_true == 0]
    fake_scores = y_pred_prob[y_true == 1]
    mean_p_real = float(np.mean(real_scores)) if len(real_scores) > 0 else 0.0
    mean_p_fake = float(np.mean(fake_scores)) if len(fake_scores) > 0 else 0.0

    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "tnr": tnr,
        "tpr": tpr,
        "mean_p_real": mean_p_real,
        "mean_p_fake": mean_p_fake,
        "score_separation": mean_p_fake - mean_p_real,
    }


def compute_score_separation(y_true, y_pred_prob):
    """Computes delta = E[P(fake)|Fake] - E[P(fake)|Real]."""
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    real_mean = np.mean(y_pred_prob[y_true == 0])
    fake_mean = np.mean(y_pred_prob[y_true == 1])
    return float(fake_mean - real_mean)


def format_benchmark_row(condition_name, baseline_acc, gated_acc, baseline_auc, gated_auc):
    """Formats markdown table row comparing baseline against dynamic gated fusion."""
    delta_acc = gated_acc - baseline_acc
    sign = "+" if delta_acc >= 0 else ""
    return (
        f"| {condition_name:<24} "
        f"| {baseline_acc:>6.2f}% | {gated_acc:>6.2f}% | **{sign}{delta_acc:>5.2f}%** "
        f"| {baseline_auc:>7.4f} | {gated_auc:>7.4f} |"
    )
