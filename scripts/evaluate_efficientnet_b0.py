"""
Evaluate Best EfficientNet-B0 Checkpoint on Validation Set and Output All Artifacts.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(r"D:\ML-PROJECT")
TORCH_CACHE_DIR = PROJECT_ROOT / "models" / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet_classifier import build_efficientnet_b0_binary, count_parameters
from src.data.polyp_classification_dataset import PolypClassificationDataset


def evaluate_and_generate_artifacts(
    checkpoint_path: str = r"D:\ML-PROJECT\outputs\models\efficientnet_b0_baseline\best_model.pth",
    output_dir: str = r"D:\ML-PROJECT\outputs\models\efficientnet_b0_baseline",
    batch_size: int = 8
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    best_epoch = checkpoint["epoch"]
    config = checkpoint.get("config", {})
    print(f"Loaded checkpoint from epoch {best_epoch}")

    model = build_efficientnet_b0_binary(pretrained=False, num_classes=2, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    param_info = count_parameters(model)

    val_ds = PolypClassificationDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    all_filepaths = []

    start_time = time.time()
    with torch.no_grad():
        for images, labels, meta in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_filenames.extend(meta["filename"])
            all_filepaths.extend(meta["filepath"])

    total_val_inference_time = time.time() - start_time
    num_samples = len(all_labels)
    latency_per_img_ms = (total_val_inference_time / max(num_samples, 1)) * 1000.0

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs_polyp = np.array([p[1] for p in all_probs])

    val_loss = running_loss / max(num_samples, 1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, probs_polyp)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Save predictions CSV
    df_preds = pd.DataFrame({
        "filename": all_filenames,
        "filepath": all_filepaths,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_non_polyp": [round(p[0], 6) for p in all_probs],
        "prob_polyp": [round(p[1], 6) for p in all_probs]
    })
    preds_csv_path = out_path / "validation_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved validation predictions to {preds_csv_path}")

    # Save metrics summary JSON
    summary = {
        "model_architecture": "EfficientNet-B0 (ImageNet-1k Pretrained, Frozen Backbone)",
        "best_epoch": int(best_epoch),
        "evaluation_split": "val (N = 320)",
        "test_quarantine_preserved": True,
        "parameters": param_info,
        "validation_metrics": {
            "val_loss": round(float(val_loss), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }
        },
        "computational_timing": {
            "validation_inference_total_s": round(total_val_inference_time, 2),
            "inference_latency_per_image_ms": round(latency_per_img_ms, 2)
        },
        "config": config
    }
    summary_path = out_path / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {summary_path}")

    # Save comparison CSV
    baseline_comp = [
        {
            "Model": "HistGradientBoosting (Handcrafted 35 Features)",
            "Accuracy": 0.7844,
            "Precision": 0.7286,
            "Recall": 0.9062,
            "Specificity": 0.6625,
            "F1-Score": 0.8078,
            "ROC-AUC": 0.8736,
            "TP": 145,
            "FP": 54,
            "TN": 106,
            "FN": 15,
            "Train Time (s)": 2.70,
            "Inference Time / Img (ms)": 0.05
        },
        {
            "Model": "EfficientNet-B0 (Pretrained Transfer Learning)",
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "Specificity": round(float(spec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Train Time (s)": 1350.0,  # ~22.5 mins total on CPU
            "Inference Time / Img (ms)": round(latency_per_img_ms, 2)
        }
    ]
    df_comp = pd.DataFrame(baseline_comp)
    comp_csv_path = out_path / "baseline_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved baseline comparison CSV to {comp_csv_path}")

    # Generate Markdown Report
    report_md_path = out_path / "EFFICIENTNET_BASELINE_REPORT.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Pretrained EfficientNet-B0 Baseline Validation Report\n\n")
        f.write("**Task:** Binary Polyp vs Non-Polyp Classification\n")
        f.write("**Architecture:** EfficientNet-B0 (ImageNet-1k Pretrained, Frozen Backbone)\n")
        f.write("**Split Source:** `config/classification_splits.json`\n")
        f.write(f"- **Training Pool:** 1,280 images (640 polyp, 640 non-polyp)\n")
        f.write(f"- **Validation Set:** 320 images (160 polyp, 160 non-polyp)\n")
        f.write(f"- **Final Test Set:** 400 images (**Strictly Quarantined & Untouched**)\n")
        f.write(f"**Best Epoch:** {best_epoch}\n\n")
        f.write("## 1. Measured Validation Performance (N = 320)\n\n")
        f.write("| Metric | EfficientNet-B0 | HistGradientBoosting Baseline | Delta |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Accuracy** | **{acc*100:.2f}%** | 78.44% | **{(acc - 0.7844)*100:+.2f}%** |\n")
        f.write(f"| **Precision** | **{prec*100:.2f}%** | 72.86% | **{(prec - 0.7286)*100:+.2f}%** |\n")
        f.write(f"| **Recall (Sensitivity)** | **{rec*100:.2f}%** | 90.62% | **{(rec - 0.9062)*100:+.2f}%** |\n")
        f.write(f"| **Specificity** | **{spec*100:.2f}%** | 66.25% | **{(spec - 0.6625)*100:+.2f}%** |\n")
        f.write(f"| **F1-Score** | **{f1:.4f}** | 0.8078 | **{f1 - 0.8078:+.4f}** |\n")
        f.write(f"| **ROC-AUC** | **{auc:.4f}** | 0.8736 | **{auc - 0.8736:+.4f}** |\n\n")
        f.write("## 2. Confusion Matrix (Validation N = 320)\n\n")
        f.write("| | Predicted Non-Polyp (0) | Predicted Polyp (1) | Total |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Actual Non-Polyp (0)** | **TN = {tn}** | **FP = {fp}** | 160 |\n")
        f.write(f"| **Actual Polyp (1)** | **FN = {fn}** | **TP = {tp}** | 160 |\n")
        f.write(f"| **Total** | {tn+fn} | {fp+tp} | 320 |\n\n")
        f.write("## 3. Computational Timing and Trade-Offs\n\n")
        f.write(f"- **EfficientNet-B0 Inference Latency:** **{latency_per_img_ms:.2f} ms / image** ({total_val_inference_time:.2f}s total for 320 images on CPU)\n")
        f.write(f"- **HistGradientBoosting Inference Latency:** ~0.05 ms / feature vector (classification only, but feature extraction pipeline takes ~1-3s per image)\n")

    print(f"Saved report to {report_md_path}")
    print("\nEVALUATION COMPLETE!")
    return summary


if __name__ == "__main__":
    evaluate_and_generate_artifacts()
