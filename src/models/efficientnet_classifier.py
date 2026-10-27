import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Ensure torch cache stays on D drive with ample free space
os.environ.setdefault("TORCH_HOME", r"D:\ML-PROJECT\models\torch_cache")


def build_efficientnet_b0_binary(
    pretrained: bool = True,
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Builds an EfficientNet-B0 binary classifier for polyp vs non-polyp detection.

    Args:
        pretrained: If True, loads official ImageNet-1k pretrained weights.
        num_classes: Output dimension (default 2: [0=non_polyp, 1=polyp]).
        freeze_backbone: If True, freezes feature extractor layers.
        dropout: Classifier dropout probability.

    Returns:
        Configured nn.Module.
    """
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace final classification head
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )

    # Ensure classifier parameters require grad
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Counts total, trainable, and frozen parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen
    }
