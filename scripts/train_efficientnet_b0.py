"""
Train and Evaluate Pretrained EfficientNet-B0 Binary Classification Baseline
Task: Binary Polyp vs Non-Polyp Classification
Preserves all project files, masks, and test set quarantine.
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

# Set TORCH_HOME before importing torch/torchvision to keep cache on D drive
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


def run_smoke_test(device: torch.device, batch_size: int = 4) -> bool:
    """Runs a minimal smoke test (8 train samples, 4 val samples) to confirm safety."""
    print("=" * 60)
    print("RUNNING SMOKE TEST")
    print(f"Device: {device}, Initial RAM: {get_ram_info()}")
    print("=" * 60)

    train_ds = PolypClassificationDataset(split="train", max_samples=8)
    val_ds = PolypClassificationDataset(split="val", max_samples=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_efficientnet_b0_binary(pretrained=True, num_classes=2, freeze_backbone=True)
    model.to(device)

    param_info = count_parameters(model)
    print(f"Model parameters: {param_info}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    t0 = time.time()
    train_loss, train_acc, step_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_res = evaluate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(f"Smoke test train step time: {step_time:.3f}s, train loss: {train_loss:.4f}, acc: {train_acc:.2f}")
    print(f"Smoke test val eval time: {val_res['inference_time_s']:.3f}s, per img: {val_res['inference_time_per_image_ms']:.1f}ms")
    print(f"Smoke test total time: {elapsed:.2f}s, Final RAM: {get_ram_info()}")
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True


def run_experiment(
    batch_size: int = 4,
    epochs: int = 10,
    patience: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: str = r"D:\ML-PROJECT\outputs\models\efficientnet_b0_baseline"
) -> Dict[str, Any]:
    """Runs full baseline training and validation evaluation."""
    set_seed(seed)
    device = torch.device("cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EFFICIENTNET-B0 BASELINE TRAINING")
    print(f"Seed: {seed}, Batch Size: {batch_size}, Max Epochs: {epochs}, Patience: {patience}, LR: {lr}")
    print(f"Initial RAM: {get_ram_info()}")
    print(f"Outputs will be saved to: {out_path}")
    print("=" * 60)

    # Instantiate datasets
    train_ds = PolypClassificationDataset(split="train")
    val_ds = PolypClassificationDataset(split="val")

    print(f"Loaded Train set: {len(train_ds)} samples")
    print(f"Loaded Val set: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model with frozen backbone
    model = build_efficientnet_b0_binary(pretrained=True, num_classes=2, freeze_backbone=True)
    model.to(device)

    param_info = count_parameters(model)
    print(f"Model parameters: {param_info}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    total_training_time = 0.0

    checkpoint_path = out_path / "best_model.pth"

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
            "val_recall": round(val_metrics["recall"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
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
            f"Recall: {val_metrics['recall']:.4f}, AUC: {val_metrics['roc_auc']:.4f} | "
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

    # Load best checkpoint for final evaluation
    print(f"\nLoading best checkpoint from epoch {best_epoch} for final validation evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_eval = evaluate(model, val_loader, criterion, device)

    # Save validation predictions CSV
    df_preds = pd.DataFrame(final_val_eval["predictions"])
    preds_csv_path = out_path / "validation_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved validation predictions to {preds_csv_path}")

    # Save training history JSON
    history_json_path = out_path / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json_path}")

    # Save metrics summary JSON
    summary = {
        "model_architecture": "EfficientNet-B0 (Pretrained ImageNet-1k, Frozen Backbone)",
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "total_training_time_s": round(total_training_time, 2),
        "total_inference_time_val_s": round(final_val_eval["inference_time_s"], 2),
        "inference_latency_per_image_ms": round(final_val_eval["inference_time_per_image_ms"], 2),
        "parameters": param_info,
        "validation_metrics": {
            "accuracy": round(final_val_eval["accuracy"], 4),
            "precision": round(final_val_eval["precision"], 4),
            "recall": round(final_val_eval["recall"], 4),
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

    # Build comparison against HistGradientBoosting baseline
    # Baseline numbers from results/v2_handcrafted_baseline_report.md
    baseline_comparison = [
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
            "Accuracy": summary["validation_metrics"]["accuracy"],
            "Precision": summary["validation_metrics"]["precision"],
            "Recall": summary["validation_metrics"]["recall"],
            "Specificity": summary["validation_metrics"]["specificity"],
            "F1-Score": summary["validation_metrics"]["f1_score"],
            "ROC-AUC": summary["validation_metrics"]["roc_auc"],
            "TP": summary["validation_metrics"]["confusion_matrix"]["tp"],
            "FP": summary["validation_metrics"]["confusion_matrix"]["fp"],
            "TN": summary["validation_metrics"]["confusion_matrix"]["tn"],
            "FN": summary["validation_metrics"]["confusion_matrix"]["fn"],
            "Train Time (s)": summary["total_training_time_s"],
            "Inference Time / Img (ms)": summary["inference_latency_per_image_ms"]
        }
    ]
    df_comp = pd.DataFrame(baseline_comparison)
    comp_csv_path = out_path / "baseline_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved baseline comparison to {comp_csv_path}")

    # Generate Markdown Report
    report_md_path = out_path / "EFFICIENTNET_BASELINE_REPORT.md"
    cm_dict = summary["validation_metrics"]["confusion_matrix"]
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# EfficientNet-B0 Baseline Validation Report\n\n")
        f.write(f"**Architecture:** EfficientNet-B0 (ImageNet-1k Pretrained, Frozen Backbone)\n")
        f.write(f"**Split Source:** `config/classification_splits.json` (1,280 Train / 320 Val / 400 Test Quarantined)\n")
        f.write(f"**Best Epoch:** {best_epoch} (Early stopping patience: {patience})\n\n")
        f.write("## 1. Validation Performance (N = 320)\n\n")
        f.write("| Metric | EfficientNet-B0 | HistGradientBoosting Baseline | Difference |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Accuracy** | {summary['validation_metrics']['accuracy']*100:.2f}% | 78.44% | {(summary['validation_metrics']['accuracy'] - 0.7844)*100:+.2f}% |\n")
        f.write(f"| **Precision** | {summary['validation_metrics']['precision']*100:.2f}% | 72.86% | {(summary['validation_metrics']['precision'] - 0.7286)*100:+.2f}% |\n")
        f.write(f"| **Recall (Sensitivity)** | {summary['validation_metrics']['recall']*100:.2f}% | 90.62% | {(summary['validation_metrics']['recall'] - 0.9062)*100:+.2f}% |\n")
        f.write(f"| **Specificity** | {summary['validation_metrics']['specificity']*100:.2f}% | 66.25% | {(summary['validation_metrics']['specificity'] - 0.6625)*100:+.2f}% |\n")
        f.write(f"| **F1-Score** | {summary['validation_metrics']['f1_score']:.4f} | 0.8078 | {summary['validation_metrics']['f1_score'] - 0.8078:+.4f} |\n")
        f.write(f"| **ROC-AUC** | {summary['validation_metrics']['roc_auc']:.4f} | 0.8736 | {summary['validation_metrics']['roc_auc'] - 0.8736:+.4f} |\n\n")
        f.write("## 2. Confusion Matrix (Validation N = 320)\n\n")
        f.write("| | Predicted Non-Polyp | Predicted Polyp |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Actual Non-Polyp (N=160)** | **TN = {cm_dict['tn']}** | **FP = {cm_dict['fp']}** |\n")
        f.write(f"| **Actual Polyp (N=160)** | **FN = {cm_dict['fn']}** | **TP = {cm_dict['tp']}** |\n\n")
        f.write("## 3. Computational Timing\n\n")
        f.write(f"- **Total Training Time:** {summary['total_training_time_s']:.2f} seconds\n")
        f.write(f"- **Validation Inference Latency:** {summary['inference_latency_per_image_ms']:.2f} ms / image ({summary['total_inference_time_val_s']:.2f}s total for 320 images)\n")
        f.write(f"- **HistGradientBoosting Inference Latency:** ~0.05 ms / feature vector (excluding active contour extraction)\n")

    print(f"Saved comprehensive report to {report_md_path}")
    print("=" * 60)
    print("ALL EXPERIMENT ARTIFACTS GENERATED SUCCESSFULLY")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Baseline")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test only")
    parser.add_argument("--output-dir", type=str, default=r"D:\ML-PROJECT\outputs\models\efficientnet_b0_baseline")

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
