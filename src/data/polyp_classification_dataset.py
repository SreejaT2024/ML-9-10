import os
import json
from typing import Optional, Callable, Dict, Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocessing.scopeguide_mask import ScopeGuideMask

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(
    split: str,
    img_size: int = 224,
    mask_scopeguide: bool = False
) -> transforms.Compose:
    """
    Returns image transformation pipelines.
    Augmentation is applied only to the training set.
    If mask_scopeguide is True, deterministic ScopeGuide occlusion is applied
    as the first step before resizing and normalization.
    """
    tfms = []
    if mask_scopeguide:
        tfms.append(ScopeGuideMask())

    if split == "train":
        tfms.extend([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif split in ("val", "test"):
        tfms.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unsupported split for transforms: {split}")

    return transforms.Compose(tfms)


class PolypClassificationDataset(Dataset):
    """
    PyTorch Dataset for binary polyp classification reading from
    config/classification_splits.json.

    Safeguards:
    - Quarantines the 400-image test set by default (raises ValueError if requested without allow_test_split=True).
    - Excludes masks and handcrafted feature CSVs.
    """

    def __init__(
        self,
        splits_json_path: str = r"D:\ML-PROJECT\config\classification_splits.json",
        project_root: str = r"D:\ML-PROJECT",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        mask_scopeguide: bool = False,
        allow_test_split: bool = False,
    ):
        if split not in ("train", "val"):
            if split == "test" and not allow_test_split:
                raise ValueError("CRITICAL: Access to quarantined test split is prohibited during development!")
            elif split == "test" and allow_test_split:
                pass
            else:
                raise ValueError(f"Invalid split '{split}'. Allowed: 'train', 'val', 'test'.")

        self.split = split
        self.project_root = project_root
        self.splits_json_path = splits_json_path
        self.mask_scopeguide = mask_scopeguide
        self.allow_test_split = allow_test_split

        with open(splits_json_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        self.class_to_idx = splits_data["metadata"]["class_to_idx"]  # non_polyp: 0, polyp: 1
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        raw_items = splits_data[split]
        if max_samples is not None:
            raw_items = raw_items[:max_samples]

        self.items: List[Dict[str, Any]] = []
        for item in raw_items:
            # Resolve absolute path
            rel_path = item["filepath"]
            full_path = os.path.join(project_root, rel_path)
            label_str = item["label"]
            label_idx = self.class_to_idx[label_str]
            self.items.append({
                "filename": item["filename"],
                "filepath": rel_path,
                "full_path": full_path,
                "label_name": label_str,
                "label": label_idx
            })

        self.transform = transform or get_default_transforms(split, mask_scopeguide=mask_scopeguide)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        meta = self.items[idx]
        image_path = meta["full_path"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor_img = self.transform(image)
        label = meta["label"]

        return tensor_img, label, {
            "filename": meta["filename"],
            "filepath": meta["filepath"],
            "label_name": meta["label_name"]
        }
