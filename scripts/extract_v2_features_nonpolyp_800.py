"""Extract Stage A V2 Features for 800 Development Non-Polyp Images.

Processes strictly the 800 development non-polyp images from config/classification_splits.json
(640 train + 160 val) using StageAFeaturePipelineV2 with unchanged parameters.

Outputs:
- outputs/features/hyperkvasir_stageA_features_v2.csv (800 rows, 39 columns)
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from src.utils.io import load_image
from src.features.feature_pipeline_v2 import (
    StageAFeaturePipelineV2,
    FEATURE_NAMES,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
)

_thread_local = threading.local()


def get_pipeline_instance() -> StageAFeaturePipelineV2:
    """Thread-local StageAFeaturePipelineV2 instance."""
    if not hasattr(_thread_local, "pipeline"):
        _thread_local.pipeline = StageAFeaturePipelineV2()
    return _thread_local.pipeline


def process_single_frame(item: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a single non-polyp frame without ground-truth access."""
    pipeline = get_pipeline_instance()
    filepath = PROJECT_ROOT / item["filepath"]
    rgb_img = load_image(filepath)
    # gt_mask is strictly None
    result = pipeline.process_sample(rgb_img, filename=item["filename"], gt_mask=None)
    row = pipeline.to_feature_row(result)
    # Also attach category and filepath for clean traceability
    row["category"] = item.get("category", "non_polyp")
    row["filepath"] = item["filepath"]
    return row


def run_nonpolyp_feature_extraction(max_workers: int = 6, batch_size: int = 25) -> Path:
    print("=" * 80)
    print("STAGE A V2: EXTRACTING FEATURES FOR 800 NON-POLYP DEVELOPMENT IMAGES")
    print("=" * 80)

    splits_path = PROJECT_ROOT / "config" / "classification_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Classification splits file not found: {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_np = [x for x in splits["train"] if x["label"] == "non_polyp"]
    val_np = [x for x in splits["val"] if x["label"] == "non_polyp"]
    dev_np = train_np + val_np

    if len(dev_np) != 800:
        raise ValueError(f"Expected exactly 800 dev non-polyp images, found {len(dev_np)}")

    print(f"\n[1] Loaded 800 development non-polyp samples from {splits_path.name}:")
    print(f"    - Training split:   {len(train_np)} samples")
    print(f"    - Validation split: {len(val_np)} samples")
    print(f"    - Quarantined test split is strictly untouched.")

    output_dir = PROJECT_ROOT / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / ".checkpoint_stageA_features_v2_nonpolyp.json"
    target_csv_path = output_dir / "hyperkvasir_stageA_features_v2.csv"

    # Resume from checkpoint if available
    results_dict: Dict[str, Dict[str, Any]] = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results_dict = {row["filepath"]: row for row in saved}
            print(f"[2] Resuming from checkpoint: {len(results_dict)}/800 samples already completed.")
        except Exception as e:
            print(f"[!] Warning: Failed to parse checkpoint ({e}). Starting fresh.")
            results_dict = {}
    else:
        print("[2] No prior checkpoint found. Starting fresh.")

    remaining_items = [item for item in dev_np if item["filepath"] not in results_dict]
    print(f"    - Remaining items: {len(remaining_items)}")
    print(f"    - Concurrency: {max_workers} worker threads (ThreadPoolExecutor)")

    start_time = time.perf_counter()
    completed_count = len(results_dict)

    if remaining_items:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for b_idx in range(0, len(remaining_items), batch_size):
                batch = remaining_items[b_idx : b_idx + batch_size]
                t_b0 = time.perf_counter()

                batch_rows = list(executor.map(process_single_frame, batch))

                for row in batch_rows:
                    results_dict[row["filepath"]] = row

                completed_count += len(batch_rows)
                dt_batch = time.perf_counter() - t_b0
                elapsed_total = time.perf_counter() - start_time
                rate = len(batch_rows) / max(dt_batch, 0.001)

                latest_sample = batch_rows[-1]
                print(
                    f"[{completed_count:03d}/800] Batch {b_idx // batch_size + 1:02d} done "
                    f"({len(batch_rows)} items in {dt_batch:.1f}s, {rate:.1f} img/s) | "
                    f"Elapsed: {elapsed_total:.1f}s | "
                    f"Latest: {latest_sample['filename'][:20]} -> {latest_sample['segmentation_status']} "
                    f"(Conf: {latest_sample['confidence']:.3f})"
                )

                # Atomically update checkpoint
                temp_cp = checkpoint_file.with_suffix(".tmp")
                with open(temp_cp, "w", encoding="utf-8") as f:
                    json.dump(list(results_dict.values()), f)
                temp_cp.replace(checkpoint_file)

    total_time = time.perf_counter() - start_time
    print(f"\n[3] Extraction complete! Processed in {total_time:.2f}s.")

    # Order results strictly by original dev_np sequence
    ordered_rows = [results_dict[item["filepath"]] for item in dev_np]
    df = pd.DataFrame(ordered_rows)

    # Standard columns for hyperkvasir_stageA_features_v2.csv
    # Include FEATURE_COLUMNS plus category and filepath
    export_cols = FEATURE_COLUMNS + ["category", "filepath"]
    df = df[export_cols]

    # Validate
    assert len(df) == 800, f"Expected 800 rows, got {len(df)}"
    assert df["filepath"].nunique() == 800, "Duplicate filepaths found!"
    assert df[FEATURE_NAMES].isna().sum().sum() == 0, "NaN values detected in feature columns!"
    assert np.isinf(df[FEATURE_NAMES].to_numpy(dtype=float)).sum() == 0, "Infinite values detected!"

    # Save final separate CSV
    df.to_csv(target_csv_path, index=False)
    print(f"\n[4] Saved non-polyp features CSV to: {target_csv_path}")
    print(f"    File size: {target_csv_path.stat().st_size / 1024:.1f} KB")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print("\n" + "=" * 80)
    print("NON-POLYP FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return target_csv_path


if __name__ == "__main__":
    run_nonpolyp_feature_extraction()
