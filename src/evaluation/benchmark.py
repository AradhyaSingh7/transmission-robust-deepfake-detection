"""
Benchmark Loader and Comparative Audit Suite
"""
import json
from pathlib import Path
from ..utils.metrics import format_benchmark_row


def load_audited_benchmarks(results_dir=None):
    """Loads audited benchmark files generated during evaluation.
    
    Returns:
        tuple (baseline_results, gated_results)
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    else:
        results_dir = Path(results_dir)

    baseline_path = results_dir / "baseline_ablation_results.json"
    gated_path = results_dir / "dynamic_gated_fusion_results.json"

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_raw = json.load(f)
        baseline_data = baseline_raw.get("results", {}).get("full", {})

    with open(gated_path, "r", encoding="utf-8") as f:
        gated_data = json.load(f)

    return baseline_data, gated_data


def run_comparative_audit(results_dir=None):
    """Generates the master comparative evaluation table corresponding to Table 1 & Table 3 in the paper."""
    baseline_data, gated_data = load_audited_benchmarks(results_dir)

    conditions_map = [
        ("Clean", "Clean"),
        ("JPEG Q=100", "JPEG Q=100"),
        ("JPEG Q=75", "JPEG Q=75"),
        ("JPEG Q=50", "JPEG Q=50"),
        ("JPEG Q=25", "JPEG Q=25"),
        ("WebP Q=75", "WebP Q=75"),
        ("WebP Q=25", "WebP Q=25"),
        ("Resize 75%", "Resize 75%"),
        ("Resize 50%", "Resize 50%"),
        ("Compound (Q50+Res75%)", "Combined (Q50+Res75%)"),
    ]

    rows = []
    for cond_name, gated_key in conditions_map:
        b_item = baseline_data.get(cond_name, {})
        g_item = gated_data.get(gated_key, {})

        b_acc = b_item.get("accuracy", 0.0)
        b_auc = b_item.get("auc", 0.0)

        g_acc = g_item.get("default_acc", 0.0)
        g_auc = g_item.get("auc", 0.0)

        row_str = format_benchmark_row(cond_name, b_acc, g_acc, b_auc, g_auc)
        rows.append(row_str)

    return rows
