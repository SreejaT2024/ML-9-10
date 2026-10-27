"""Execution Script for 20-Image Development-Set Segmentation Pilot (v1 vs v2).

Evaluates baseline v1 ActiveContourSegmenter vs remediated v2 ActiveContourSegmenterV2
strictly on a deterministic 20-sample slice from the Kvasir-SEG development split.
Ground-truth masks are strictly quarantined and loaded only during evaluation.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.utils.io import load_image, load_mask
from src.preprocessing.image_preprocess import PolypImagePreprocessor
from src.segmentation.active_contour import ActiveContourSegmenter
from src.segmentation.active_contour_v2 import ActiveContourSegmenterV2
from src.segmentation.metrics import evaluate_segmentation
from src.data.splits import get_splits


def draw_contour_overlay(
    rgb_image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draws ground truth (green) and predicted (pred_color) boundary contours on RGB image."""
    overlay = rgb_image.copy()

    # Draw Ground Truth in Green (0, 255, 0)
    gt_contours, _ = cv2.findContours(
        (gt_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)

    # Draw Prediction contour
    if np.count_nonzero(pred_mask) > 0:
        pred_contours, _ = cv2.findContours(
            (pred_mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, pred_contours, -1, pred_color, 2)

    return overlay


def run_pilot():
    print("=" * 80)
    print("STARTING 20-IMAGE DEVELOPMENT-SET SEGMENTATION PILOT (v1 vs v2)")
    print("=" * 80)

    config = get_config()
    images_dir = Path(config.paths.kvasir_images_dir)
    masks_dir = Path(config.paths.kvasir_masks_dir)

    output_dir = PROJECT_ROOT / "results" / "v2_pilot_20"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Deterministic Selection of 20 images from Development Split
    dev_files, eval_files = get_splits()
    print(f"Total Dev images available: {len(dev_files)}")
    print(f"Total Eval images quarantined: {len(eval_files)}")

    # Sample 20 evenly-spaced items across the 800-image development list for broad representation
    step = len(dev_files) // 20
    pilot_filenames = [dev_files[i * step] for i in range(20)]

    filenames_file = output_dir / "pilot_20_filenames.txt"
    with open(filenames_file, "w", encoding="utf-8") as f:
        for fn in pilot_filenames:
            f.write(f"{fn}\n")
    print(f"Selected 20 development samples saved to: {filenames_file}")

    # Initialize Modules
    preprocessor = PolypImagePreprocessor()
    segmenter_v1 = ActiveContourSegmenter(config=config)
    segmenter_v2 = ActiveContourSegmenterV2(config=config)

    records = []
    contact_data = []

    for idx, fn in enumerate(pilot_filenames):
        img_path = images_dir / fn
        mask_path = masks_dir / fn

        print(f"[{idx+1:02d}/20] Processing {fn}...")

        # Load RGB image
        rgb_img = load_image(img_path)
        h, w = rgb_img.shape[:2]
        total_pixels = h * w

        # Common Preprocessing
        prep_img = preprocessor.preprocess_for_segmentation(rgb_img)

        # ------------------------------------------------------------------
        # RUN BASELINE v1 (Zero Ground Truth Access)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        v1_mask, v1_success, v1_info = segmenter_v1.segment(prep_img, rgb_image=rgb_img)
        v1_runtime = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # RUN REMEDIATED v2 (Zero Ground Truth Access)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        v2_mask, v2_success, v2_info = segmenter_v2.segment(prep_img, rgb_image=rgb_img)
        v2_runtime = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # EVALUATION LAYER (Ground Truth accessed strictly here)
        # ------------------------------------------------------------------
        gt_mask = load_mask(mask_path)
        gt_fg = np.count_nonzero(gt_mask == 255)
        gt_area_ratio = gt_fg / float(total_pixels)

        # Calculate Segmentation Metrics
        m1 = evaluate_segmentation(v1_mask, gt_mask)
        m2 = evaluate_segmentation(v2_mask, gt_mask)

        v1_fg = np.count_nonzero(v1_mask == 255)
        v2_fg = np.count_nonzero(v2_mask == 255)
        v1_area_ratio = v1_fg / float(total_pixels)
        v2_area_ratio = v2_fg / float(total_pixels)

        dice_diff = m2["dice"] - m1["dice"]

        rec = {
            "index": idx + 1,
            "filename": fn,
            "gt_area_ratio": float(gt_area_ratio),
            "v1_dice": float(m1["dice"]),
            "v1_iou": float(m1["iou"]),
            "v1_precision": float(m1["precision"]),
            "v1_recall": float(m1["recall"]),
            "v1_area_ratio": float(v1_area_ratio),
            "v1_success": bool(v1_success),
            "v1_runtime_sec": float(v1_runtime),
            "v2_dice": float(m2["dice"]),
            "v2_iou": float(m2["iou"]),
            "v2_precision": float(m2["precision"]),
            "v2_recall": float(m2["recall"]),
            "v2_area_ratio": float(v2_area_ratio),
            "v2_success": bool(v2_success),
            "v2_status": str(v2_info.get("status", "unknown")),
            "v2_confidence": float(v2_info.get("confidence", 0.0)),
            "v2_runtime_sec": float(v2_runtime),
            "dice_diff": float(dice_diff),
        }
        records.append(rec)

        print(
            f"       v1 Dice: {m1['dice']:.4f} (Prec: {m1['precision']:.3f}, Rec: {m1['recall']:.3f}) | "
            f"v2 Dice: {m2['dice']:.4f} (Prec: {m2['precision']:.3f}, Rec: {m2['recall']:.3f}) | "
            f"Diff: {dice_diff:+.4f} | v2 Status: {v2_info.get('status')}"
        )

        # ------------------------------------------------------------------
        # GENERATE PER-IMAGE VISUALIZATION (6 subplots)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Sample {idx+1:02d}: {fn}\nv1 Dice: {m1['dice']:.4f} | v2 Dice: {m2['dice']:.4f} (Diff: {dice_diff:+.4f})",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Original Image
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title("1. Original Image")
        axes[0, 0].axis("off")

        # 2. Ground Truth Mask
        axes[0, 1].imshow(gt_mask, cmap="gray")
        axes[0, 1].set_title(f"2. Ground Truth (Area: {gt_area_ratio*100:.1f}%)")
        axes[0, 1].axis("off")

        # 3. v1 Predicted Mask
        axes[0, 2].imshow(v1_mask, cmap="gray")
        axes[0, 2].set_title(
            f"3. v1 Mask (Dice: {m1['dice']:.3f}, Prec: {m1['precision']:.2f}, Rec: {m1['recall']:.2f})"
        )
        axes[0, 2].axis("off")

        # 4. v2 Predicted Mask
        axes[1, 0].imshow(v2_mask, cmap="gray")
        axes[1, 0].set_title(
            f"4. v2 Mask (Dice: {m2['dice']:.3f}, Prec: {m2['precision']:.2f}, Rec: {m2['recall']:.2f})\nStatus: {v2_info.get('status')} (Conf: {v2_info.get('confidence', 0.0):.2f})"
        )
        axes[1, 0].axis("off")

        # 5. v1 Boundary Overlay (Green = GT, Red = v1)
        v1_overlay = draw_contour_overlay(rgb_img, gt_mask, v1_mask, pred_color=(255, 0, 0))
        axes[1, 1].imshow(v1_overlay)
        axes[1, 1].set_title("5. v1 Overlay (Green: GT, Red: v1)")
        axes[1, 1].axis("off")

        # 6. v2 Boundary Overlay (Green = GT, Blue = v2)
        v2_overlay = draw_contour_overlay(rgb_img, gt_mask, v2_mask, pred_color=(0, 120, 255))
        axes[1, 2].imshow(v2_overlay)
        axes[1, 2].set_title("6. v2 Overlay (Green: GT, Blue: v2)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        stem = Path(fn).stem
        vis_path = vis_dir / f"{idx+1:02d}_{stem}.png"
        fig.savefig(vis_path, dpi=120)
        plt.close(fig)

        # Store for contact sheet
        contact_data.append({
            "stem": stem,
            "img": v2_overlay,
            "v1_dice": m1["dice"],
            "v2_dice": m2["dice"],
            "diff": dice_diff,
            "status": v2_info.get("status"),
        })

    # Save Metrics CSV
    df = pd.DataFrame(records)
    csv_path = output_dir / "pilot_20_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_path}")

    # ------------------------------------------------------------------
    # GENERATE 20-IMAGE CONTACT SHEET (4 rows x 5 columns)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.suptitle(
        "Kvasir-SEG Development-Set Pilot (20 Samples)\nOverlay: Green = Ground Truth, Blue = v2 Prediction",
        fontsize=16,
        fontweight="bold",
    )

    for i, data in enumerate(contact_data):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        ax.imshow(data["img"])
        color = "green" if data["diff"] > 0 else ("red" if data["diff"] < -0.05 else "black")
        ax.set_title(
            f"#{i+1:02d}: {data['stem'][:12]}..\nv1: {data['v1_dice']:.3f} | v2: {data['v2_dice']:.3f}\nDiff: {data['diff']:+.3f} [{data['status']}]",
            fontsize=9,
            color=color,
        )
        ax.axis("off")

    plt.tight_layout()
    contact_path = output_dir / "pilot_contact_sheet.png"
    fig.savefig(contact_path, dpi=120)
    plt.close(fig)
    print(f"Saved contact sheet to: {contact_path}")

    # ------------------------------------------------------------------
    # GENERATE COMPREHENSIVE SUMMARY REPORT
    # ------------------------------------------------------------------
    v1_d = df["v1_dice"]
    v2_d = df["v2_dice"]
    v1_iou = df["v1_iou"]
    v2_iou = df["v2_iou"]
    v1_prec = df["v1_precision"]
    v2_prec = df["v2_precision"]
    v1_rec = df["v1_recall"]
    v2_rec = df["v2_recall"]

    n_improved = int((df["dice_diff"] > 0.01).sum())
    n_worse = int((df["dice_diff"] < -0.01).sum())
    n_same = int((df["dice_diff"].abs() <= 0.01).sum())

    mean_diff = float(df["dice_diff"].mean())

    best_v2_idx = df["v2_dice"].idxmax()
    worst_v2_idx = df["v2_dice"].idxmin()
    largest_imp_idx = df["dice_diff"].idxmax()
    largest_reg_idx = df["dice_diff"].idxmin()

    # Rejection analysis
    rejected_df = df[~df["v2_success"] | (df["v2_status"] != "valid_candidate")]

    summary_md = f"""# KVASIR-SEG 20-IMAGE DEVELOPMENT-SET PILOT REPORT

**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Scope:** Strictly 20 evenly-spaced samples from the 800-image development split (`splits_v2.json['dev']`).  
**Quarantine Confirmation:** The 200-image held-out evaluation set was 100% quarantined. Zero evaluation images were accessed.  

---

## 1. EXECUTIVE SUMMARY & METRIC COMPARISON

| Metric | Baseline v1 | Remediated v2 | Delta (v2 - v1) |
|---|---|---|---|
| **Mean Dice** | **{v1_d.mean():.4f}** | **{v2_d.mean():.4f}** | **{v2_d.mean() - v1_d.mean():+.4f}** |
| **Median Dice** | {v1_d.median():.4f} | {v2_d.median():.4f} | {v2_d.median() - v1_d.median():+.4f} |
| **Std Dev Dice** | {v1_d.std():.4f} | {v2_d.std():.4f} | {v2_d.std() - v1_d.std():+.4f} |
| **Min Dice** | {v1_d.min():.4f} | {v2_d.min():.4f} | {v2_d.min() - v1_d.min():+.4f} |
| **Max Dice** | {v1_d.max():.4f} | {v2_d.max():.4f} | {v2_d.max() - v1_d.max():+.4f} |
| **Mean IoU** | {v1_iou.mean():.4f} | {v2_iou.mean():.4f} | {v2_iou.mean() - v1_iou.mean():+.4f} |
| **Median IoU** | {v1_iou.median():.4f} | {v2_iou.median():.4f} | {v2_iou.median() - v1_iou.median():+.4f} |
| **Mean Precision** | **{v1_prec.mean():.4f}** | **{v2_prec.mean():.4f}** | **{v2_prec.mean() - v1_prec.mean():+.4f}** |
| **Mean Recall** | **{v1_rec.mean():.4f}** | **{v2_rec.mean():.4f}** | **{v2_rec.mean() - v1_rec.mean():+.4f}** |
| **Mean Runtime (sec/img)** | {df['v1_runtime_sec'].mean():.3f}s | {df['v2_runtime_sec'].mean():.3f}s | {df['v2_runtime_sec'].mean() - df['v1_runtime_sec'].mean():+.3f}s |

### Distribution Benchmarks:
- **Dice $\ge$ 0.5:** v1 = `{(v1_d >= 0.5).sum()}/20` | v2 = `{(v2_d >= 0.5).sum()}/20`
- **Dice $\ge$ 0.7:** v1 = `{(v1_d >= 0.7).sum()}/20` | v2 = `{(v2_d >= 0.7).sum()}/20`
- **Dice $<$ 0.3:** v1 = `{(v1_d < 0.3).sum()}/20` | v2 = `{(v2_d < 0.3).sum()}/20`

### Pairwise Comparison:
- **Images with v2 Dice > v1 Dice:** **{n_improved} / 20** ({n_improved/20*100:.1f}%)
- **Images with v1 Dice > v2 Dice:** **{n_worse} / 20** ({n_worse/20*100:.1f}%)
- **Images Approximately Unchanged:** **{n_same} / 20** ({n_same/20*100:.1f}%)
- **Mean Pairwise Improvement:** **{mean_diff:+.4f} Dice**

---

## 2. AUTOMATIC CASE CATEGORIZATION

- **Best v2 Result:** Sample #{df.loc[best_v2_idx, 'index']:02d} (`{df.loc[best_v2_idx, 'filename']}`) — **Dice = {df.loc[best_v2_idx, 'v2_dice']:.4f}** (v1 Dice: {df.loc[best_v2_idx, 'v1_dice']:.4f})
- **Worst v2 Result:** Sample #{df.loc[worst_v2_idx, 'index']:02d} (`{df.loc[worst_v2_idx, 'filename']}`) — **Dice = {df.loc[worst_v2_idx, 'v2_dice']:.4f}** (v1 Dice: {df.loc[worst_v2_idx, 'v1_dice']:.4f})
- **Largest Improvement:** Sample #{df.loc[largest_imp_idx, 'index']:02d} (`{df.loc[largest_imp_idx, 'filename']}`) — **Gain: {df.loc[largest_imp_idx, 'dice_diff']:+.4f}** (v1: {df.loc[largest_imp_idx, 'v1_dice']:.4f} $\to$ v2: {df.loc[largest_imp_idx, 'v2_dice']:.4f})
- **Largest Regression:** Sample #{df.loc[largest_reg_idx, 'index']:02d} (`{df.loc[largest_reg_idx, 'filename']}`) — **Loss: {df.loc[largest_reg_idx, 'dice_diff']:+.4f}** (v1: {df.loc[largest_reg_idx, 'v1_dice']:.4f} $\to$ v2: {df.loc[largest_reg_idx, 'v2_dice']:.4f})
- **Typical / Median v2 Sample:** Sample #{df.loc[(v2_d - v2_d.median()).abs().idxmin(), 'index']:02d} (`{df.loc[(v2_d - v2_d.median()).abs().idxmin(), 'filename']}`) — **v2 Dice: {v2_d.median():.4f}**

---

## 3. REJECTION & LOW-CONFIDENCE ANALYSIS

Total Rejections in 20 images: **{len(rejected_df)}**

"""
    if len(rejected_df) == 0:
        summary_md += "No samples were rejected in this 20-image pilot. All 20 images yielded valid polyp candidates.\n"
    else:
        summary_md += "| # | Filename | v2 Status | Confidence | GT Area % | v1 Dice | v2 Dice | Probable Reason |\n"
        summary_md += "|---|---|---|---|---|---|---|---|\n"
        for _, r in rejected_df.iterrows():
            summary_md += f"| {int(r['index']):02d} | `{r['filename']}` | {r['v2_status']} | {r['v2_confidence']:.2f} | {r['gt_area_ratio']*100:.1f}% | {r['v1_dice']:.3f} | {r['v2_dice']:.3f} | Rejection threshold ({r['v2_confidence']:.2f} < 0.30) |\n"

    summary_md += f"""
---

## 4. SANITY CHECKS & FAILURE MODE FREQUENCY

- **Mask Covers Most of Image (> 55% frame):** v1 = {(df['v1_area_ratio'] > 0.55).sum()}/20 | v2 = {(df['v2_area_ratio'] > 0.55).sum()}/20
- **Tiny Meaningless Mask (< 0.5% frame):** v1 = {(df['v1_area_ratio'] < 0.005).sum()}/20 | v2 = {(df['v2_area_ratio'] < 0.005).sum()}/20
- **Severe Mucosal Leakage (Precision < 20%):** v1 = {(v1_prec < 0.20).sum()}/20 | v2 = {(v2_prec < 0.20).sum()}/20
- **Excessive Empty-Mask Rejections:** {len(rejected_df)} / 20 ({len(rejected_df)/20*100:.1f}%)

---

## 5. COMPLETE PER-SAMPLE RESULTS

| # | Filename | GT Area% | v1 Dice | v2 Dice | Delta | v1 Prec | v2 Prec | v1 Rec | v2 Rec | v2 Status |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df.iterrows():
        summary_md += f"| {int(r['index']):02d} | `{r['filename']}` | {r['gt_area_ratio']*100:.1f}% | {r['v1_dice']:.3f} | {r['v2_dice']:.3f} | {r['dice_diff']:+.3f} | {r['v1_precision']:.3f} | {r['v2_precision']:.3f} | {r['v1_recall']:.3f} | {r['v2_recall']:.3f} | {r['v2_status']} |\n"

    summary_file = output_dir / "pilot_20_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nSaved comprehensive pilot report to: {summary_file}")
    print("\nPILOT COMPLETE.")


if __name__ == "__main__":
    run_pilot()
