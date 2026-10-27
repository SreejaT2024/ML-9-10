"""Rerun exact 20-image development pilot with integrated production ActiveContourSegmenterV2 (B1).

Compares:
- Old B0 (current production before B1)
- Original B1 experiment
- Integrated Production B1

Generates metrics, comparisons, and visual regression panels for:
- Case 03, Case 12, Case 18, Case 20 (Recovered cases)
- Case 02, Case 07, Case 11 (False-positive / rejected cases)
- Case 17, Case 19 (Accepted-but-wrong cases)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
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
from src.segmentation.active_contour_v2 import ActiveContourSegmenterV2
from src.segmentation.metrics import evaluate_segmentation


def draw_comparison_overlay(
    rgb_image: np.ndarray,
    gt_mask: np.ndarray,
    b0_mask: np.ndarray,
    b1_mask: np.ndarray,
) -> np.ndarray:
    """Draws Ground Truth (Green), B0 Mask (Red), and Integrated B1 Mask (Cyan) contours."""
    overlay = rgb_image.copy()

    # Ground Truth: Green (0, 255, 0)
    gt_cnt, _ = cv2.findContours(
        (gt_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, gt_cnt, -1, (0, 255, 0), 2)

    # B0 Mask: Red (255, 0, 0)
    if np.count_nonzero(b0_mask) > 0:
        b0_cnt, _ = cv2.findContours(
            (b0_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, b0_cnt, -1, (255, 0, 0), 2)

    # Integrated B1 Mask: Cyan (0, 200, 255)
    if np.count_nonzero(b1_mask) > 0:
        b1_cnt, _ = cv2.findContours(
            (b1_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, b1_cnt, -1, (0, 200, 255), 2)

    return overlay


def main():
    print("=" * 80)
    print("RUNNING 20-IMAGE PILOT ON INTEGRATED PRODUCTION V2 (B1 VIGNETTE-AWARE)")
    print("=" * 80)

    config = get_config()
    images_dir = Path(config.paths.kvasir_images_dir)
    masks_dir = Path(config.paths.kvasir_masks_dir)

    output_dir = PROJECT_ROOT / "results" / "v2_integrated_b1_pilot"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pilot_fn_file = PROJECT_ROOT / "results" / "v2_pilot_20" / "pilot_20_filenames.txt"
    with open(pilot_fn_file, "r", encoding="utf-8") as f:
        pilot_filenames = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(pilot_filenames)} exact pilot samples from {pilot_fn_file.name}.")

    preprocessor = PolypImagePreprocessor()
    segmenter = ActiveContourSegmenterV2()

    # Load B0 masks from prior experiment for exact visual comparison
    b0_exp_file = PROJECT_ROOT / "results" / "v2_vignette_analysis" / "per_image_comparison.csv"
    b0_info_df = pd.read_csv(b0_exp_file) if b0_exp_file.exists() else None

    # Load B0 masks from results/v2_vignette_analysis/visualizations or recompute if needed
    # We can reconstruct or run B0 directly to have the exact mask arrays for overlays
    # Let's run integrated production segmenter
    results = []
    masks_b1 = {}
    runtimes = []

    print("\nExecuting integrated ActiveContourSegmenterV2.segment()...")
    total_start = time.perf_counter()

    for idx, fn in enumerate(pilot_filenames):
        img_path = images_dir / fn
        mask_path = masks_dir / fn

        rgb_img = load_image(img_path)
        gt_mask = load_mask(mask_path)
        prep_img = preprocessor.preprocess_for_segmentation(rgb_img)

        t0 = time.perf_counter()
        pred_mask, success, info = segmenter.segment(prep_img, rgb_image=rgb_img)
        dt = time.perf_counter() - t0
        runtimes.append(dt)

        masks_b1[fn] = pred_mask

        gt_fg = int(np.count_nonzero(gt_mask > 127))
        pred_fg = int(np.count_nonzero(pred_mask > 127))

        if pred_fg > 0:
            eval_metrics = evaluate_segmentation(pred_mask, gt_mask)
            status = "Accepted"
        else:
            eval_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            status = "Rejected"

        results.append({
            "idx": idx + 1,
            "filename": fn,
            "status": status,
            "success": success,
            "confidence": float(info.get("confidence", 0.0)),
            "border_penalty": float(info.get("p_border", 0.0)),
            "dice": float(eval_metrics["dice"]),
            "iou": float(eval_metrics["iou"]),
            "precision": float(eval_metrics["precision"]),
            "recall": float(eval_metrics["recall"]),
            "pred_pixels": pred_fg,
            "gt_pixels": gt_fg,
            "runtime_sec": dt,
            "centroid": str(info.get("centroid", (0, 0))),
            "info_status": info.get("status", ""),
        })

        print(
            f"[{idx+1:02d}/20] {fn[:24]} | Status: {status:8s} | "
            f"Score: {info.get('confidence', 0.0):.4f} | BorderPen: {info.get('p_border', 0.0):.1f} | "
            f"Dice: {eval_metrics['dice']:.4f} | Runtime: {dt:.2f}s"
        )

    total_time = time.perf_counter() - total_start
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "integrated_b1_pilot_metrics.csv", index=False)

    # Compute Summary Metrics
    dice_vals = df_results["dice"].values
    iou_vals = df_results["iou"].values
    prec_vals = df_results["precision"].values
    rec_vals = df_results["recall"].values
    pred_areas = df_results["pred_pixels"].values

    accepted_df = df_results[df_results["status"] == "Accepted"]
    rejected_count = int(np.count_nonzero(df_results["status"] == "Rejected"))
    accepted_dice_0 = int(np.count_nonzero((df_results["status"] == "Accepted") & (df_results["dice"] == 0.0)))

    summary_metrics = {
        "Mean Dice": float(np.mean(dice_vals)),
        "Median Dice": float(np.median(dice_vals)),
        "Std Dice": float(np.std(dice_vals)),
        "Mean IoU": float(np.mean(iou_vals)),
        "Mean Precision": float(np.mean(prec_vals)),
        "Mean Recall": float(np.mean(rec_vals)),
        "Rejected count": rejected_count,
        "Accepted count": len(accepted_df),
        "Accepted Dice=0 count": accepted_dice_0,
        "Mean predicted area (all 20)": float(np.mean(pred_areas)),
        "Mean predicted area (accepted)": float(np.mean(accepted_df["pred_pixels"].values)) if len(accepted_df) > 0 else 0.0,
        "Total Runtime (sec)": float(total_time),
        "Mean Runtime per image (sec)": float(np.mean(runtimes)),
    }

    print("\n" + "=" * 80)
    print("INTEGRATED B1 PRODUCTION RESULTS SUMMARY")
    print("=" * 80)
    for k, v in summary_metrics.items():
        if "count" in k or "count" in k.lower():
            print(f"  {k:32s}: {int(v)}")
        elif "Runtime" in k or "area" in k:
            print(f"  {k:32s}: {v:.2f}")
        else:
            print(f"  {k:32s}: {v:.4f}")

    # Load B0 results from per_image_comparison.csv
    comp_df = pd.read_csv(PROJECT_ROOT / "results" / "v2_vignette_analysis" / "per_image_comparison.csv")
    b0_dice = comp_df["B0_Dice"].values
    b1_exp_dice = comp_df["B1_Dice"].values

    comparison_table = pd.DataFrame({
        "idx": df_results["idx"],
        "filename": df_results["filename"],
        "B0_Status": comp_df["B0_Status"],
        "B0_Score": comp_df["B0_Score"],
        "B0_BorderPen": comp_df["B0_BorderPen"],
        "B0_Dice": comp_df["B0_Dice"],
        "B1_Exp_Status": comp_df["B1_Status"],
        "B1_Exp_Score": comp_df["B1_Score"],
        "B1_Exp_BorderPen": comp_df["B1_BorderPen"],
        "B1_Exp_Dice": comp_df["B1_Dice"],
        "Integrated_Status": df_results["status"],
        "Integrated_Score": df_results["confidence"],
        "Integrated_BorderPen": df_results["border_penalty"],
        "Integrated_Dice": df_results["dice"],
        "Dice_Diff_vs_B1_Exp": df_results["dice"] - comp_df["B1_Dice"],
    })
    comparison_table.to_csv(output_dir / "b0_vs_b1exp_vs_integrated_b1.csv", index=False)

    print("\n" + "=" * 80)
    print("NUMERICAL VERIFICATION: INTEGRATED B1 vs EXPERIMENTAL B1")
    print("=" * 80)
    max_dice_diff = float(np.max(np.abs(df_results["dice"].values - b1_exp_dice)))
    max_score_diff = float(np.max(np.abs(df_results["confidence"].values - comp_df["B1_Score"].values)))
    print(f"Max absolute Dice discrepancy: {max_dice_diff:.8f}")
    print(f"Max absolute Score discrepancy: {max_score_diff:.8f}")

    # Generate Visual Comparisons for the 9 key cases
    # Cases: 03, 12, 18, 20 (Recovered), 02, 07, 11 (False-positives/rejections), 17, 19 (Accepted-but-wrong)
    target_indices = [3, 12, 18, 20, 2, 7, 11, 17, 19]

    # Reconstruct B0 masks for visual comparison overlays
    print("\nGenerating visual comparison panels for key cases...")
    for idx_1based in target_indices:
        i = idx_1based - 1
        fn = pilot_filenames[i]
        rgb = load_image(images_dir / fn)
        gt = load_mask(masks_dir / fn)
        prep = preprocessor.preprocess_for_segmentation(rgb)

        m_b1 = masks_b1[fn]

        # Re-run B0 on this image for exact visual overlay
        # B0: raw morphgac mask without FOV mask clipping, crude border mask
        # We can run segmenter with border_mask clipping for B0 overlay:
        # Or load B0 mask from v2_pilot_20 or reconstruct
        # Let's reconstruct exact B0 mask
        h, w = rgb.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            proc_sub = cv2.resize(prep, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_sub = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
            proc_sub, rgb_sub = prep, rgb

        inp_rgb = segmenter.inpaint_specular_highlights(rgb_sub)
        sal = segmenter.compute_saliency_map(inp_rgb)
        sm = segmenter.generate_seeds(sal, rgb_image=inp_rgb)
        if np.count_nonzero(sm) > 0:
            inp_g = cv2.cvtColor(inp_rgb, cv2.COLOR_RGB2GRAY)
            gac_m = segmenter.evolve_morphgac(inp_g, sm)
            # Old B0 score components logic:
            crude_b = segmenter.get_black_border_mask(rgb_sub)
            n_l, lbls, stats, c_roids = cv2.connectedComponentsWithStats((gac_m > 0).astype(np.uint8), 8)
            b0_cands = []
            for lb in range(1, n_l):
                ar = int(stats[lb, cv2.CC_STAT_AREA])
                ar_ratio = ar / float(new_h * new_w)
                if not (segmenter.min_component_area_ratio <= ar_ratio <= segmenter.max_component_area_ratio):
                    continue
                cb = (lbls == lb).astype(np.uint8)
                t_edge = bool(
                    np.any(cb[0, :]) or np.any(cb[new_h - 1, :]) or np.any(cb[:, 0]) or np.any(cb[:, new_w - 1])
                    or np.any(cb[crude_b == 0])
                )
                p_b = 1.0 if t_edge else 0.0
                cx, cy = float(c_roids[lb][0]), float(c_roids[lb][1])
                s_ar = float(np.clip(np.log(ar + 1) / np.log(0.15 * new_h * new_w + 1), 0.0, 1.0))
                # Contrast
                dil = cv2.dilate(cb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                er = cv2.erode(cb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                o_rim = (dil > 0) & (cb == 0) & (crude_b > 0)
                i_rim = (cb > 0) & (er == 0)
                if np.any(i_rim) and np.any(o_rim):
                    c_val = abs(float(np.mean(inp_g[i_rim])) - float(np.mean(inp_g[o_rim])))
                    s_diff = max(0.0, float(np.mean(sal[cb > 0])) - float(np.mean(sal[o_rim])))
                else:
                    c_val, s_diff = 0.0, 0.0
                s_c = float(np.clip((c_val / 35.0) * 0.7 + (s_diff / 0.5) * 0.3, 0.0, 1.0))
                d_cnt = np.hypot(cx - new_w / 2.0, cy - new_h / 2.0)
                s_cen = float(np.clip(1.0 - (d_cnt / (0.5 * np.hypot(new_w, new_h))), 0.0, 1.0))
                sc = float(np.clip(0.25 * s_ar + 0.35 * s_c + 0.15 * s_cen - 0.25 * p_b, 0.0, 1.0))
                b0_cands.append({"comp_mask": cb, "score": sc, "p_border": p_b})
            b0_cands.sort(key=lambda x: x["score"], reverse=True)
            if b0_cands and b0_cands[0]["score"] >= 0.30:
                best_b0 = b0_cands[0]["comp_mask"] * 255
                cnts_b0, _ = cv2.findContours(best_b0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fb0 = np.zeros_like(best_b0)
                if cnts_b0:
                    cv2.drawContours(fb0, cnts_b0, -1, 255, thickness=cv2.FILLED)
                    best_b0 = fb0
                best_b0 = cv2.morphologyEx(best_b0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                if (new_h, new_w) != (h, w):
                    m_b0 = cv2.resize(best_b0, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    m_b0 = best_b0
                m_b0 = cv2.bitwise_and(m_b0, segmenter.get_black_border_mask(rgb))
                if np.count_nonzero(m_b0) / float(h * w) > 0.55:
                    m_b0 = np.zeros((h, w), dtype=np.uint8)
            else:
                m_b0 = np.zeros((h, w), dtype=np.uint8)
        else:
            m_b0 = np.zeros((h, w), dtype=np.uint8)

        # Plot 4-panel comparison
        row_b0 = comp_df.iloc[i]
        row_b1 = df_results.iloc[i]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        overlay = draw_comparison_overlay(rgb, gt, m_b0, m_b1)

        axes[0].imshow(rgb)
        axes[0].set_title(f"Case {idx_1based:02d}: Original Image\n{fn[:24]}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title(f"Ground Truth\nArea: {row_b1['gt_pixels']:,} px", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(m_b1, cmap="gray")
        axes[2].set_title(
            f"Integrated B1 Mask ({row_b1['status']})\nDice: {row_b1['dice']:.4f} | Score: {row_b1['confidence']:.3f}",
            fontsize=10,
        )
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title(
            f"Contour Overlay\nGT (Green) | B0 (Red) | B1 (Cyan)",
            fontsize=10,
        )
        axes[3].axis("off")

        # Add legend
        patch_gt = mpatches.Patch(color="lime", label="Ground Truth")
        patch_b0 = mpatches.Patch(color="red", label=f"Old B0 (Dice={row_b0['B0_Dice']:.3f})")
        patch_b1 = mpatches.Patch(color="cyan", label=f"Integrated B1 (Dice={row_b1['dice']:.3f})")
        axes[3].legend(handles=[patch_gt, patch_b0, patch_b1], loc="lower right", fontsize=8)

        plt.tight_layout()
        vis_filename = vis_dir / f"case_{idx_1based:02d}_{fn.replace('.jpg', '')}_comparison.png"
        plt.savefig(vis_filename, dpi=120)
        plt.close()
        print(f"Saved visualization: {vis_filename.name}")

    print("\nVisualizations complete! All 9 key cases generated under:")
    print(f"  {vis_dir}")


if __name__ == "__main__":
    main()
