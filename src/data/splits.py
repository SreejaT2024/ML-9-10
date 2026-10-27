"""Reproducible Dataset Split Module for Kvasir-SEG (Development vs Held-out Evaluation).

Generates a deterministic 80% development set (800 samples) and 20% held-out
evaluation set (200 samples) using a fixed random seed and alphabetically sorted filenames.
Ground-truth masks in the held-out evaluation set are strictly quarantined and never
used for hyperparameter tuning.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config


SPLITS_FILE_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "splits_v2.json"


def generate_dev_eval_splits(
    seed: int = 42,
    dev_ratio: float = 0.80,
    splits_file: Optional[Path] = None,
    force_recreate: bool = False,
) -> Dict[str, List[str]]:
    """Generates or loads reproducible 80/20 development/evaluation split.

    Args:
        seed: Random seed for reproducible shuffling.
        dev_ratio: Fraction of dataset allocated to development set (default: 0.80).
        splits_file: Path to save/load JSON splits file.
        force_recreate: If True, regenerates the split even if file exists.

    Returns:
        Dict with keys 'dev' and 'eval', each containing list of image filenames.
    """
    save_path = Path(splits_file) if splits_file else SPLITS_FILE_PATH

    if save_path.is_file() and not force_recreate:
        with open(save_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        return splits_data

    config = get_config()
    images_dir = Path(config.paths.kvasir_images_dir)

    all_filenames = sorted(
        [
            f.name
            for f in images_dir.iterdir()
            if f.suffix.lower() in config.dataset.supported_extensions
        ]
    )

    if len(all_filenames) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    total_count = len(all_filenames)
    dev_count = int(round(total_count * dev_ratio))

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(total_count)

    dev_indices = sorted(shuffled_indices[:dev_count])
    eval_indices = sorted(shuffled_indices[dev_count:])

    dev_filenames = [all_filenames[i] for i in dev_indices]
    eval_filenames = [all_filenames[i] for i in eval_indices]

    splits_data = {
        "metadata": {
            "dataset": "Kvasir-SEG",
            "seed": seed,
            "dev_ratio": dev_ratio,
            "total_count": total_count,
            "dev_count": len(dev_filenames),
            "eval_count": len(eval_filenames),
        },
        "dev": dev_filenames,
        "eval": eval_filenames,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    return splits_data


def get_splits() -> Tuple[List[str], List[str]]:
    """Convenience accessor for development and evaluation filenames.

    Returns:
        Tuple of (dev_filenames, eval_filenames).
    """
    splits_data = generate_dev_eval_splits()
    return splits_data["dev"], splits_data["eval"]


if __name__ == "__main__":
    splits = generate_dev_eval_splits(force_recreate=True)
    print(f"Generated Kvasir-SEG split:")
    print(f"  Dev set: {len(splits['dev'])} images ({splits['metadata']['dev_ratio']*100:.1f}%)")
    print(f"  Eval set: {len(splits['eval'])} images ({(1-splits['metadata']['dev_ratio'])*100:.1f}%)")
    print(f"  Saved to: {SPLITS_FILE_PATH}")
