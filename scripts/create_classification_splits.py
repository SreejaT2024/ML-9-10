"""Create and validate dataset index for binary polyp vs non-polyp classification.

Preserves existing Kvasir-SEG splits from config/splits_v2.json:
- 800 development polyp images split into 640 train / 160 val.
- 200 evaluation polyp images quarantined strictly for test.
- 1,000 non-polyp images sampled from approved HyperKvasir lower-GI categories:
  * lower-gi-tract/anatomical-landmarks/cecum
  * lower-gi-tract/anatomical-landmarks/retroflex-rectum
  * lower-gi-tract/quality-of-mucosal-views/bbps-2-3
- Deterministically allocated into 640 train / 160 val / 200 test.

Generates: config/classification_splits.json
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_and_validate_classification_splits(
    seed: int = 42,
    output_path: Path = PROJECT_ROOT / "config" / "classification_splits.json",
) -> Dict[str, Any]:
    print("=" * 80)
    print("BUILDING AND VALIDATING BINARY CLASSIFICATION DATASET INDEX")
    print("=" * 80)

    # 1. Load existing splits_v2.json
    splits_v2_path = PROJECT_ROOT / "config" / "splits_v2.json"
    if not splits_v2_path.exists():
        raise FileNotFoundError(f"splits_v2.json not found at: {splits_v2_path}")

    with open(splits_v2_path, "r", encoding="utf-8") as f:
        splits_v2 = json.load(f)

    dev_polyps: List[str] = splits_v2.get("dev", [])
    eval_polyps: List[str] = splits_v2.get("eval", [])

    if len(dev_polyps) != 800:
        raise ValueError(f"Expected 800 dev polyp images in splits_v2.json, found {len(dev_polyps)}")
    if len(eval_polyps) != 200:
        raise ValueError(f"Expected 200 eval polyp images in splits_v2.json, found {len(eval_polyps)}")

    kvasir_images_dir = PROJECT_ROOT / "kvasir-seg" / "Kvasir-SEG" / "images"

    # Verify all polyp images exist
    for fn in dev_polyps + eval_polyps:
        p = kvasir_images_dir / fn
        if not p.is_file():
            raise FileNotFoundError(f"Kvasir-SEG image missing: {p}")

    print(f"\n[1] Verified Kvasir-SEG polyp images:")
    print(f"    - Development set: {len(dev_polyps)} images")
    print(f"    - Quarantined evaluation set: {len(eval_polyps)} images")

    # Split 800 dev polyps into 640 train / 160 val using deterministic permutation
    rng_polyp = np.random.RandomState(seed)
    perm_dev = rng_polyp.permutation(len(dev_polyps))
    train_polyp_fns = [dev_polyps[i] for i in sorted(perm_dev[:640])]
    val_polyp_fns = [dev_polyps[i] for i in sorted(perm_dev[640:])]
    test_polyp_fns = list(eval_polyps)

    # 2. Collect HyperKvasir approved non-polyp images
    approved_cats = {
        "cecum": PROJECT_ROOT / "hyper-kvasir-labeled-images" / "labeled-images" / "lower-gi-tract" / "anatomical-landmarks" / "cecum",
        "retroflex-rectum": PROJECT_ROOT / "hyper-kvasir-labeled-images" / "labeled-images" / "lower-gi-tract" / "anatomical-landmarks" / "retroflex-rectum",
        "bbps-2-3": PROJECT_ROOT / "hyper-kvasir-labeled-images" / "labeled-images" / "lower-gi-tract" / "quality-of-mucosal-views" / "bbps-2-3",
    }

    # Verify directory existence
    for cat_name, cat_dir in approved_cats.items():
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"Approved category directory missing: {cat_dir}")

    # Proportional allocations for 1,000 non-polyp samples across the 3 categories
    # Cecum (1,009): 396 total -> 254 train, 63 val, 79 test
    # Retroflex-Rectum (391): 153 total -> 98 train, 24 val, 31 test
    # BBPS 2-3 (1,148): 451 total -> 288 train, 73 val, 90 test
    allocations = {
        "cecum": {"total": 396, "train": 254, "val": 63, "test": 79},
        "retroflex-rectum": {"total": 153, "train": 98, "val": 24, "test": 31},
        "bbps-2-3": {"total": 451, "train": 288, "val": 73, "test": 90},
    }

    train_nonpolyp_items = []
    val_nonpolyp_items = []
    test_nonpolyp_items = []

    print(f"\n[2] Sampling approved HyperKvasir Lower-GI non-polyp categories (seed={seed}):")
    for cat_name, cat_dir in approved_cats.items():
        all_files = sorted([f.name for f in cat_dir.glob("*.jpg") if f.is_file()])
        alloc = allocations[cat_name]
        print(f"    - {cat_name:<18}: {len(all_files)} available -> sampling {alloc['total']} "
              f"({alloc['train']} train, {alloc['val']} val, {alloc['test']} test)")

        cat_rng = np.random.RandomState(seed + int(hashlib.md5(cat_name.encode()).hexdigest()[:8], 16) % 10000)
        shuffled = cat_rng.permutation(len(all_files))
        selected_idx = shuffled[:alloc["total"]]

        t_idx = selected_idx[:alloc["train"]]
        v_idx = selected_idx[alloc["train"] : alloc["train"] + alloc["val"]]
        te_idx = selected_idx[alloc["train"] + alloc["val"] : alloc["total"]]

        for i in sorted(t_idx):
            train_nonpolyp_items.append((cat_name, all_files[i]))
        for i in sorted(v_idx):
            val_nonpolyp_items.append((cat_name, all_files[i]))
        for i in sorted(te_idx):
            test_nonpolyp_items.append((cat_name, all_files[i]))

    # Helper function to construct sample record
    def make_polyp_record(fn: str) -> Dict[str, Any]:
        rel_path = f"kvasir-seg/Kvasir-SEG/images/{fn}"
        return {
            "filepath": rel_path,
            "filename": fn,
            "label": "polyp",
            "label_idx": 1,
            "source_dataset": "Kvasir-SEG",
            "category": "polyp",
        }

    def make_nonpolyp_record(cat_name: str, fn: str) -> Dict[str, Any]:
        if cat_name in {"cecum", "retroflex-rectum"}:
            rel_path = f"hyper-kvasir-labeled-images/labeled-images/lower-gi-tract/anatomical-landmarks/{cat_name}/{fn}"
        else:
            rel_path = f"hyper-kvasir-labeled-images/labeled-images/lower-gi-tract/quality-of-mucosal-views/{cat_name}/{fn}"
        return {
            "filepath": rel_path,
            "filename": fn,
            "label": "non_polyp",
            "label_idx": 0,
            "source_dataset": "HyperKvasir",
            "category": cat_name,
        }

    # Construct split lists
    train_list = [make_polyp_record(fn) for fn in train_polyp_fns] + [
        make_nonpolyp_record(cat, fn) for cat, fn in train_nonpolyp_items
    ]
    val_list = [make_polyp_record(fn) for fn in val_polyp_fns] + [
        make_nonpolyp_record(cat, fn) for cat, fn in val_nonpolyp_items
    ]
    test_list = [make_polyp_record(fn) for fn in test_polyp_fns] + [
        make_nonpolyp_record(cat, fn) for cat, fn in test_nonpolyp_items
    ]

    # Sort each split deterministically by filepath
    train_list.sort(key=lambda x: x["filepath"])
    val_list.sort(key=lambda x: x["filepath"])
    test_list.sort(key=lambda x: x["filepath"])

    # ── VALIDATION CHECKS BEFORE SAVING ──
    print("\n[3] Executing Mandatory Pre-Saving Validations:")

    # Check 1: Exact image counts for every class and split
    counts = {
        "train": {
            "polyp": sum(1 for x in train_list if x["label"] == "polyp"),
            "non_polyp": sum(1 for x in train_list if x["label"] == "non_polyp"),
            "total": len(train_list),
        },
        "val": {
            "polyp": sum(1 for x in val_list if x["label"] == "polyp"),
            "non_polyp": sum(1 for x in val_list if x["label"] == "non_polyp"),
            "total": len(val_list),
        },
        "test": {
            "polyp": sum(1 for x in test_list if x["label"] == "polyp"),
            "non_polyp": sum(1 for x in test_list if x["label"] == "non_polyp"),
            "total": len(test_list),
        },
    }

    assert counts["train"]["polyp"] == 640, f"Expected 640 train polyps, got {counts['train']['polyp']}"
    assert counts["train"]["non_polyp"] == 640, f"Expected 640 train non-polyps, got {counts['train']['non_polyp']}"
    assert counts["val"]["polyp"] == 160, f"Expected 160 val polyps, got {counts['val']['polyp']}"
    assert counts["val"]["non_polyp"] == 160, f"Expected 160 val non-polyps, got {counts['val']['non_polyp']}"
    assert counts["test"]["polyp"] == 200, f"Expected 200 test polyps, got {counts['test']['polyp']}"
    assert counts["test"]["non_polyp"] == 200, f"Expected 200 test non-polyps, got {counts['test']['non_polyp']}"
    print(f"    [PASS] Exact Counts: Train={counts['train']['total']} (640/640), "
          f"Val={counts['val']['total']} (160/160), Test={counts['test']['total']} (200/200), "
          f"Total=2,000 images (1,000/1,000).")

    # Check 2: All image paths exist on disk
    all_records = train_list + val_list + test_list
    for rec in all_records:
        full_p = PROJECT_ROOT / rec["filepath"]
        if not full_p.is_file():
            raise FileNotFoundError(f"Path does not exist on disk: {full_p}")
    print(f"    [PASS] Physical Disk Integrity: All 2,000 image filepaths verified to exist.")

    # Check 3: No duplicate paths across splits
    all_paths = [rec["filepath"] for rec in all_records]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError(f"Duplicate paths found across splits! {len(all_paths)} total vs {len(set(all_paths))} unique.")
    print(f"    [PASS] Path Mutual Exclusivity: 2,000 unique paths, 0 duplicates across splits.")

    # Check 4: No exact-content MD5 duplicates between positive and negative images
    print("    Checking MD5 content hashes between positive and negative images...")
    positive_hashes = {}
    negative_hashes = {}

    for rec in all_records:
        full_p = PROJECT_ROOT / rec["filepath"]
        h = hashlib.md5(full_p.read_bytes()).hexdigest()
        if rec["label"] == "polyp":
            positive_hashes[h] = rec["filepath"]
        else:
            negative_hashes[h] = rec["filepath"]

    overlap = set(positive_hashes.keys()) & set(negative_hashes.keys())
    if overlap:
        sample_h = next(iter(overlap))
        raise ValueError(
            f"Content hash collision between positive and negative classes!\n"
            f"Polyp: {positive_hashes[sample_h]} <---> Non-polyp: {negative_hashes[sample_h]}"
        )
    print(f"    [PASS] Cross-Class Content Isolation: 0 hash collisions between polyp and non-polyp sets.")

    # Check 5: Quarantined 200 evaluation polyp images strictly excluded from train and val
    train_val_polyp_fns = set(train_polyp_fns) | set(val_polyp_fns)
    quarantine_breach = train_val_polyp_fns & set(eval_polyps)
    if quarantine_breach:
        raise ValueError(f"Quarantine breach! Evaluation images found in train/val: {quarantine_breach}")

    test_polyp_fns_set = set(test_polyp_fns)
    if test_polyp_fns_set != set(eval_polyps):
        raise ValueError("Test polyp images do not match evaluation images from splits_v2.json!")
    print(f"    [PASS] Quarantine Integrity: 100% of 200 eval polyp images remain strictly in test split.")

    # Check 6: All negative images belong to approved categories only
    approved_cat_names = set(approved_cats.keys())
    for rec in all_records:
        if rec["label"] == "non_polyp":
            if rec["category"] not in approved_cat_names:
                raise ValueError(f"Unapproved negative category found: {rec['category']}")
            if "upper-gi-tract" in rec["filepath"]:
                raise ValueError(f"Upper-GI image found in non-polyp set: {rec['filepath']}")
            if "polyps" in rec["filepath"]:
                raise ValueError(f"Polyp image found in non-polyp set: {rec['filepath']}")
    print(f"    [PASS] Category Compliance: 100% of negative images belong strictly to approved lower-GI classes.")

    # Construct final data structure
    splits_data = {
        "metadata": {
            "task": "binary_polyp_classification",
            "classes": ["non_polyp", "polyp"],
            "class_to_idx": {"non_polyp": 0, "polyp": 1},
            "seed": seed,
            "counts": counts,
            "approved_negative_sources": list(approved_cats.keys()),
            "preserves_kvasir_seg_splits_v2": True,
        },
        "train": train_list,
        "val": val_list,
        "test": test_list,
    }

    # 4. Save JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    print(f"\n[4] Saved dataset index to: {output_path}")

    # Check 7: Verify written JSON can be reloaded
    with open(output_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert len(reloaded["train"]) == 1280
    assert len(reloaded["val"]) == 320
    assert len(reloaded["test"]) == 400
    print(f"    [PASS] JSON Reload Validation: File parsed successfully and matches expected structure.")

    print("\n" + "=" * 80)
    print("ALL VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)

    return splits_data


if __name__ == "__main__":
    build_and_validate_classification_splits()
