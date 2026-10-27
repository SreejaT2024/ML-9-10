"""
Train and Evaluate ScopeGuide-Mitigated EfficientNet-B0 Binary Classification Experiment.

Task: Binary Polyp vs Non-Polyp Classification with ScopeGuide Occlusion Mitigation.
Strict Safeguards:
- Preserves original model, baseline artifacts, dataset, and classification_splits.json.
- Strictly quarantines the 400-image test set.
- Pre-training RAM and runtime estimation checks.
- Applies deterministic, documented ScopeGuide corner masking [0.65*h:h, 0:0.35*w]
  to both training and validation splits prior to resizing and normalization.
- Class-agnostic masking: identical masking applied regardless of class label.
- Training-only augmentations restricted to training split.
- Fixed seed 42, frozen EfficientNet-B0 backbone, Adam lr=1e-3, batch size 8, max 10 epochs, patience 3.
- Saves all artifacts to outputs/models/efficientnet_b0_masked/.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import psutil
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


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ram_info() -> Dict[str, float]:
    """Returns available and used system RAM in GB."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "percent_used": vm.percent
    }


def check_and_estimate_resources(
    num_train: int,
    num_val: int,
    batch_size: int = 8,
    max_epochs: int = 10,
    min_required_ram_gb: float = 0.30
) -> Dict[str, Any]:
    """
    Estimates runtime and memory footprint before training launches.
    Aborts if available system resources are insufficient.
    """
    ram = get_ram_info()
    avail_ram_gb = ram["available_gb"]

    # CPU benchmark: baseline ran ~125 seconds per epoch on this 12-core CPU with batch size 8
    # 1,280 train images = 160 batches; 320 val images = 40 batches
    est_sec_per_batch = 0.65  # forward + backward (head only) on frozen backbone
    est_epoch_train_sec = (num_train / batch_size) * est_sec_per_batch
    est_epoch_val_sec = (num_val / batch_size) * 0.45
    est_sec_per_epoch = est_epoch_train_sec + est_epoch_val_sec
    est_total_runtime_sec = est_sec_per_epoch * max_epochs
    est_total_runtime_min = est_total_runtime_sec / 60.0

    print("=" * 65)
    print("PRE-TRAINING SYSTEM & RESOURCE FEASIBILITY AUDIT")
    print("=" * 65)
    print(f"System RAM Total:     {ram['total_gb']} GB")
    print(f"System RAM Available: {avail_ram_gb} GB ({100 - ram['percent_used']:.1f}% free)")
    print(f"Logical CPU Cores:    {psutil.cpu_count(logical=True)}")
    print(f"Train Dataset Size:   {num_train} images ({num_train // batch_size} batches of size {batch_size})")
    print(f"Val Dataset Size:     {num_val} images ({num_val // batch_size} batches of size {batch_size})")
    print(f"Estimated Epoch Time: ~{est_sec_per_epoch:.1f} seconds (~{est_sec_per_epoch / 60.0:.2f} min)")
    print(f"Estimated Max Time:   ~{est_total_runtime_min:.1f} minutes for {max_epochs} epochs")
    print(f"Estimated Peak RAM:   ~0.25 GB (Frozen backbone, batch size {batch_size})")
    print("=" * 65)

    if avail_ram_gb < min_required_ram_gb:
        err_msg = (
            f"INSUFFICIENT SYSTEM RAM: Available RAM ({avail_ram_gb} GB) is below "
            f"the safety threshold ({min_required_ram_gb} GB). Aborting job to prevent OOM crash."
        )
        print(f"CRITICAL ERROR: {err_msg}")
        raise MemoryError(err_msg)

    print("Resource check PASSED: Resources are sufficient to execute the experiment.\n")
    return {
        "ram_info": ram,
        "estimated_sec_per_epoch": est_sec_per_epoch,
        "estimated_total_runtime_min": est_total_runtime_min
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Trains model for one epoch. Returns (avg_loss, accuracy, elapsed_time)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    elapsed = time.time() - start_time
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates model on validation set with full metrics and timing."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    all_filepaths = []

    start_time = time.time()

    for images, labels, meta in dataloader:
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

    inference_time = time.time() - start_time
    num_samples = len(all_labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs_polyp = np.array([p[1] for p in all_probs])

    avg_loss = running_loss / max(num_samples, 1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs_polyp)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    predictions_records = []
    for fn_name, fp_name, yt, yp, prob_pair in zip(all_filenames, all_filepaths, y_true, y_pred, all_probs):
        predictions_records.append({
            "filename": fn_name,
            "filepath": fp_name,
            "y_true": int(yt),
            "y_pred": int(yp),
            "prob_non_polyp": float(prob_pair[0]),
            "prob_polyp": float(prob_pair[1])
        })

    return {
        "val_loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "num_samples": num_samples,
        "inference_time_s": inference_time,
        "inference_time_per_image_ms": (inference_time / max(num_samples, 1)) * 1000.0,
        "predictions": predictions_records
    }


def run_smoke_test(device: torch.device, batch_size: int = 8) -> bool:
    """Runs a minimal smoke test (8 train, 8 val) to confirm end-to-end functionality."""
    print("=" * 65)
    print("RUNNING SCOPEGUIDE-MASKED SMOKE TEST")
    print(f"Device: {device}, Initial RAM: {get_ram_info()}")
    print("=" * 65)

    train_ds = PolypClassificationDataset(split="train", max_samples=8, mask_scopeguide=True)
    val_ds = PolypClassificationDataset(split="val", max_samples=8, mask_scopeguide=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_efficientnet_b0_binary(pretrained=True, num_classes=2, freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    t0 = time.time()
    train_loss, train_acc, step_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_res = evaluate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(f"Smoke test step: train loss={train_loss:.4f}, acc={train_acc:.2f}, time={step_time:.2f}s")
    print(f"Smoke test val: val loss={val_res['val_loss']:.4f}, acc={val_res['accuracy']:.2f}")
    print(f"Smoke test complete in {elapsed:.2f}s. Smoke test PASSED!")
    print("=" * 65)
    return True


def run_experiment(
    batch_size: int = 8,
    epochs: int = 10,
    patience: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: str = r"D:\ML-PROJECT\outputs\models\efficientnet_b0_masked"
) -> Dict[str, Any]:
    """Executes full ScopeGuide-mitigated training and validation evaluation."""
    set_seed(seed)
    device = torch.device("cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate datasets with ScopeGuide masking enabled
    print("\n[Step 1] Loading Datasets with ScopeGuide Occlusion Masking...")
    train_ds = PolypClassificationDataset(split="train", mask_scopeguide=True)
    val_ds = PolypClassificationDataset(split="val", mask_scopeguide=True)

    num_train = len(train_ds)
    num_val = len(val_ds)
    print(f"Loaded Train set: {num_train} samples (ScopeGuide masked before resize/augmentation)")
    print(f"Loaded Val set:   {num_val} samples (ScopeGuide masked before resize)")

    # 2. Feasibility and Resource Check
    res_audit = check_and_estimate_resources(
        num_train=num_train,
        num_val=num_val,
        batch_size=batch_size,
        max_epochs=epochs
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model construction with frozen backbone
    print("\n[Step 2] Initializing EfficientNet-B0 with Frozen Backbone...")
    model = build_efficientnet_b0_binary(pretrained=True, num_classes=2, freeze_backbone=True)
    model.to(device)
    param_info = count_parameters(model)
    print(f"Parameters: Total={param_info['total_parameters']:,}, Trainable={param_info['trainable_parameters']:,}, Frozen={param_info['frozen_parameters']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    total_training_time = 0.0

    checkpoint_path = out_path / "best_model.pth"

    print("\n[Step 3] Commencing Training Loop (Max Epochs: 10, Patience: 3)...")
    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        total_training_time += train_time

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_elapsed = time.time() - epoch_start
        ram = get_ram_info()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "confusion_matrix": val_metrics["confusion_matrix"],
            "train_time_s": round(train_time, 2),
            "val_inference_time_s": round(val_metrics["inference_time_s"], 2),
            "epoch_time_s": round(epoch_elapsed, 2),
            "available_ram_gb": ram["available_gb"]
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"Sens: {val_metrics['recall']:.4f}, Spec: {val_metrics['specificity']:.4f}, "
            f"AUC: {val_metrics['roc_auc']:.4f} | "
            f"Time: {epoch_elapsed:.1f}s | RAM Free: {ram['available_gb']} GB"
        )

        # Early stopping based on validation loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "architecture": "efficientnet_b0",
                    "experiment": "scopeguide_masked",
                    "mask_strategy": "corner [0.65*h:h, 0:0.35*w]",
                    "pretrained": True,
                    "num_classes": 2,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed
                }
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  --> Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    overall_elapsed = time.time() - training_start_time

    # 4. Load best checkpoint for final validation evaluation
    print(f"\n[Step 4] Loading best checkpoint from epoch {best_epoch} for final evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_eval = evaluate(model, val_loader, criterion, device)

    # 5. Save validation predictions CSV
    df_preds = pd.DataFrame(final_val_eval["predictions"])
    preds_csv_path = out_path / "validation_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved validation predictions to {preds_csv_path}")

    # 6. Save training history JSON
    history_json_path = out_path / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json_path}")

    # 7. Save standalone confusion matrix JSON
    cm_path = out_path / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(final_val_eval["confusion_matrix"], f, indent=2)
    print(f"Saved confusion matrix to {cm_path}")

    # 8. Save metrics summary JSON
    summary = {
        "experiment_name": "EfficientNet-B0 ScopeGuide-Mitigated Retraining",
        "model_architecture": "EfficientNet-B0 (Pretrained ImageNet-1k, Frozen Backbone)",
        "scopeguide_mitigation": {
            "mask_relative_y": [0.65, 1.0],
            "mask_relative_x": [0.0, 0.35],
            "fill_value": 0,
            "timing": "Applied before resizing and normalization",
            "class_agnostic": True,
            "splits_masked": ["train", "val"]
        },
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "total_wall_clock_time_s": round(overall_elapsed, 2),
        "total_training_time_s": round(total_training_time, 2),
        "total_inference_time_val_s": round(final_val_eval["inference_time_s"], 2),
        "inference_latency_per_image_ms": round(final_val_eval["inference_time_per_image_ms"], 2),
        "parameters": param_info,
        "validation_metrics": {
            "val_loss": round(final_val_eval["val_loss"], 4),
            "accuracy": round(final_val_eval["accuracy"], 4),
            "precision": round(final_val_eval["precision"], 4),
            "recall_sensitivity": round(final_val_eval["recall"], 4),
            "specificity": round(final_val_eval["specificity"], 4),
            "f1_score": round(final_val_eval["f1"], 4),
            "roc_auc": round(final_val_eval["roc_auc"], 4),
            "confusion_matrix": final_val_eval["confusion_matrix"]
        },
        "config": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": seed,
            "patience": patience,
            "max_epochs": epochs
        }
    }
    summary_path = out_path / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to {summary_path}")

    # 9. Load Original Baseline Metrics for Direct Comparison
    baseline_summary_path = PROJECT_ROOT / "outputs" / "models" / "efficientnet_b0_baseline" / "metrics_summary.json"
    with open(baseline_summary_path, "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    base_m = base_meta["validation_metrics"]
    base_cm = base_m["confusion_matrix"]

    masked_m = summary["validation_metrics"]
    masked_cm = masked_m["confusion_matrix"]

    # Comparative table
    comparison_data = [
        {
            "Metric": "Accuracy",
            "Original Baseline": f"{base_m['accuracy']*100:.2f}%",
            "ScopeGuide-Masked Model": f"{masked_m['accuracy']*100:.2f}%",
            "Absolute Delta": f"{(masked_m['accuracy'] - base_m['accuracy'])*100:+.2f}%"
        },
        {
            "Metric": "Precision",
            "Original Baseline": f"{base_m['precision']*100:.2f}%",
            "ScopeGuide-Masked Model": f"{masked_m['precision']*100:.2f}%",
            "Absolute Delta": f"{(masked_m['precision'] - base_m['precision'])*100:+.2f}%"
        },
        {
            "Metric": "Sensitivity (Recall)",
            "Original Baseline": f"{base_m['recall']*100:.2f}%",
            "ScopeGuide-Masked Model": f"{masked_m['recall_sensitivity']*100:.2f}%",
            "Absolute Delta": f"{(masked_m['recall_sensitivity'] - base_m['recall'])*100:+.2f}%"
        },
        {
            "Metric": "Specificity",
            "Original Baseline": f"{base_m['specificity']*100:.2f}%",
            "ScopeGuide-Masked Model": f"{masked_m['specificity']*100:.2f}%",
            "Absolute Delta": f"{(masked_m['specificity'] - base_m['specificity'])*100:+.2f}%"
        },
        {
            "Metric": "F1-Score",
            "Original Baseline": f"{base_m['f1_score']:.4f}",
            "ScopeGuide-Masked Model": f"{masked_m['f1_score']:.4f}",
            "Absolute Delta": f"{masked_m['f1_score'] - base_m['f1_score']:+.4f}"
        },
        {
            "Metric": "ROC-AUC",
            "Original Baseline": f"{base_m['roc_auc']:.4f}",
            "ScopeGuide-Masked Model": f"{masked_m['roc_auc']:.4f}",
            "Absolute Delta": f"{masked_m['roc_auc'] - base_m['roc_auc']:+.4f}"
        },
        {
            "Metric": "False Positives (FP)",
            "Original Baseline": f"{base_cm['fp']}",
            "ScopeGuide-Masked Model": f"{masked_cm['fp']}",
            "Absolute Delta": f"{masked_cm['fp'] - base_cm['fp']:+d}"
        },
        {
            "Metric": "True Positives (TP)",
            "Original Baseline": f"{base_cm['tp']}",
            "ScopeGuide-Masked Model": f"{masked_cm['tp']}",
            "Absolute Delta": f"{masked_cm['tp'] - base_cm['tp']:+d}"
        },
        {
            "Metric": "True Negatives (TN)",
            "Original Baseline": f"{base_cm['tn']}",
            "ScopeGuide-Masked Model": f"{masked_cm['tn']}",
            "Absolute Delta": f"{masked_cm['tn'] - base_cm['tn']:+d}"
        },
        {
            "Metric": "False Negatives (FN)",
            "Original Baseline": f"{base_cm['fn']}",
            "ScopeGuide-Masked Model": f"{masked_cm['fn']}",
            "Absolute Delta": f"{masked_cm['fn'] - base_cm['fn']:+d}"
        }
    ]
    df_comp = pd.DataFrame(comparison_data)
    comp_csv_path = out_path / "baseline_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved baseline comparison to {comp_csv_path}")

    # 10. Generate Comprehensive Experiment Report
    report_md_path = out_path / "SCOPE_GUIDE_MITIGATED_REPORT.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# ScopeGuide-Mitigated EfficientNet-B0 Experiment Report\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(
            "This experiment implements a separate, ScopeGuide-mitigated EfficientNet-B0 training run "
            "to evaluate model behavior when the Olympus ScopeGuide indicator is occluded during both training "
            "and validation. The original baseline (`outputs/models/efficientnet_b0_baseline/`), original dataset, "
            "`config/classification_splits.json`, and the quarantined 400-image test set were completely preserved.\n\n"
        )
        f.write("### Mitigation Methodology:\n")
        f.write(
            "- **Consistent, Documented Mask:** Based on empirical diagnostic findings, the bottom-left corner region "
            "`[y: 0.65*h to h, x: 0 to 0.35*w]` was zeroed to neutral black `(0, 0, 0)`.\n"
            "- **Class-Agnostic & Deterministic:** Applied unconditionally to all images regardless of class label (polyp and non-polyp alike).\n"
            "- **Applied Before Resizing & Normalization:** Applied to the raw un-resized image, ensuring the relative geometry aligns strictly with the ScopeGuide footprint.\n"
            "- **Augmentations Restricted:** Training-only augmentations (flips, rotation, color jitter) remained restricted to the training split.\n"
            "- **Identical Protocol:** Fixed seed 42, frozen ImageNet-pretrained backbone, Adam lr=1e-3, batch size 8, max 10 epochs with early stopping patience 3.\n\n"
        )
        f.write("## 2. Quantitative Metric Comparison (Validation Set, N = 320)\n\n")
        f.write("| Metric | Original Baseline | ScopeGuide-Masked Model | Absolute Delta |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for row in comparison_data:
            f.write(f"| **{row['Metric']}** | {row['Original Baseline']} | {row['ScopeGuide-Masked Model']} | {row['Absolute Delta']} |\n")
        f.write("\n")

        f.write("## 3. Confusion Matrix Breakdown\n\n")
        f.write("### ScopeGuide-Masked Confusion Matrix (N = 320)\n\n")
        f.write("| | Predicted Non-Polyp | Predicted Polyp |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Actual Non-Polyp (N=160)** | **TN = {masked_cm['tn']}** | **FP = {masked_cm['fp']}** |\n")
        f.write(f"| **Actual Polyp (N=160)** | **FN = {masked_cm['fn']}** | **TP = {masked_cm['tp']}** |\n\n")

        f.write("### Original Baseline Confusion Matrix (N = 320)\n\n")
        f.write("| | Predicted Non-Polyp | Predicted Polyp |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Actual Non-Polyp (N=160)** | **TN = {base_cm['tn']}** | **FP = {base_cm['fp']}** |\n")
        f.write(f"| **Actual Polyp (N=160)** | **FN = {base_cm['fn']}** | **TP = {base_cm['tp']}** |\n\n")

        f.write("## 4. Training History & Convergence Analysis\n\n")
        f.write(f"- **Best Epoch:** Epoch {best_epoch} (Validation Loss: {best_val_loss:.4f})\n")
        f.write(f"- **Total Epochs Trained:** {len(history)} / {epochs}\n")
        f.write(f"- **Total Training Time:** {summary['total_training_time_s']:.2f} seconds ({summary['total_training_time_s']/60:.2f} min)\n")
        f.write(f"- **Validation Inference Latency:** {summary['inference_latency_per_image_ms']:.2f} ms / image\n\n")

        f.write("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Sens | Val Spec | Val AUC | Epoch Time |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for h in history:
            f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']*100:.2f}% | {h['val_loss']:.4f} | {h['val_accuracy']*100:.2f}% | {h['val_recall']*100:.2f}% | {h['val_specificity']*100:.2f}% | {h['val_roc_auc']:.4f} | {h['epoch_time_s']:.1f}s |\n")
        f.write("\n")

        f.write("## 5. Critical Analysis of Remaining Dataset Limitations\n\n")
        f.write(
            "> [!IMPORTANT]\n"
            "> **Masking the ScopeGuide sub-window does NOT eliminate dataset bias.**\n"
            "> While corner occlusion prevents the model from relying directly on the high-contrast green coil graphic, "
            "> fundamental collection and domain confounders remain between the polyp and non-polyp classes:\n\n"
        )
        f.write("1. **Disparate Class Origins & Acquisition Conditions:**\n")
        f.write("   - Polyp images were acquired during targeted polypectomy or diagnostic surveillance procedures where polyps were identified, featuring specific camera focus, insufflation, and mucosal vascular patterns.\n")
        f.write("   - Non-polyp images were curated from specific anatomical reference folders in HyperKvasir (e.g., cecum landmarks, ileocecal valve, and BBPS bowel preparation scoring), introducing systematic differences in background mucosa texture, lighting, and camera equipment.\n\n")
        f.write("2. **Correlated Endoscopic Confounders:**\n")
        f.write("   - The presence of the Olympus ScopeGuide was an artifact of the specific colonoscope models utilized in certain procedure rooms. Other unmasked visual features (e.g., video processor color temperature, border vignette characteristics, and bowel lumen aspect ratios) correlate with acquisition hardware.\n\n")
        f.write("3. **Input Distribution Distortion (Occlusion Artifact):**\n")
        f.write("   - Inserting an artificial black bounding block into the bottom-left corner forces the neural network to adapt to an unnatural zero-pixel patch. Although training with this mask prevents test-time distribution shock, the black rectangle remains an out-of-distribution artifact compared to real-world clean endoscopic video.\n\n")
        f.write("## 6. Generated Experiment Artifacts\n\n")
        f.write(f"- **Model Checkpoint:** `{out_path / 'best_model.pth'}`\n")
        f.write(f"- **Training History JSON:** `{history_json_path}`\n")
        f.write(f"- **Validation Predictions CSV:** `{preds_csv_path}`\n")
        f.write(f"- **Metrics Summary JSON:** `{summary_path}`\n")
        f.write(f"- **Confusion Matrix JSON:** `{cm_path}`\n")
        f.write(f"- **Baseline Comparison CSV:** `{comp_csv_path}`\n")
        f.write(f"- **Experiment Report:** `{report_md_path}`\n")

    print(f"Saved comprehensive experiment report to {report_md_path}")
    print("=" * 65)
    print("EXPERIMENT SUCCESSFULLY COMPLETED")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ScopeGuide-Mitigated EfficientNet-B0")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test only")
    parser.add_argument("--output-dir", type=str, default=r"D:\ML-PROJECT\outputs\models\efficientnet_b0_masked")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(device=torch.device("cpu"), batch_size=args.batch_size)
    else:
        run_experiment(
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            seed=args.seed,
            output_dir=args.output_dir
        )
