"""Named model-size and optimizer settings; see fidelity notes in docs."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from .model import PlutoConfig


@dataclass(frozen=True)
class Recipe:
    model: PlutoConfig
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    min_learning_rate: float = 1e-6


def get_recipe(name):
    if name == "smoke":
        return Recipe(
            PlutoConfig(
                dim=16,
                heads=4,
                encoder_layers=1,
                decoder_layers=1,
                modes=3,
                future_steps=4,
                dropout=0.1,
            ),
            1e-3,
            1e-2,
            0,
        )
    if name == "practical":
        return Recipe(
            PlutoConfig(
                dim=64,
                heads=4,
                encoder_layers=2,
                decoder_layers=2,
                modes=6,
                future_steps=80,
                dropout=0.1,
            ),
            1e-3,
            1e-4,
            3,
        )
    if name == "paper_shape":
        return Recipe(
            PlutoConfig(
                dim=128,
                heads=4,
                encoder_layers=4,
                decoder_layers=4,
                modes=12,
                future_steps=80,
                dropout=0.1,
            ),
            1e-3,
            1e-4,
            3,
        )
    raise ValueError(f"unknown recipe: {name}")


def make_optimizer(model, learning_rate, weight_decay):
    """AdamW with the released trainer's module-specific decay exclusions.

    Biases, normalization/embedding weights and free learned parameters do not
    decay. Only Linear, convolution, attention and recurrent weights decay.
    Sorted names keep optimizer state ordering deterministic across resume.
    """
    decay_modules = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
        nn.LSTM,
        nn.GRU,
    )
    no_decay_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.Embedding,
    )
    groups = [
        {"params": [], "param_names": [], "weight_decay": weight_decay},
        {"params": [], "param_names": [], "weight_decay": 0.0},
    ]
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        module_name, _, local_name = name.rpartition(".")
        module = model.get_submodule(module_name)
        if "bias" in local_name or "weight" not in local_name:
            index = 1
        elif isinstance(module, decay_modules):
            index = 0
        elif isinstance(module, no_decay_modules):
            index = 1
        else:
            raise ValueError(f"unclassified optimizer weight: {name}")
        groups[index]["params"].append(parameter)
        groups[index]["param_names"].append(name)
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr=1e-6):
    """Epoch-based released warmup/cosine formula, clamped after its horizon."""
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup must be nonnegative and shorter than total epochs")
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    if (
        not math.isfinite(min_lr)
        or min_lr < 0
        or any(not math.isfinite(lr) or lr <= 0 or min_lr > lr for lr in base_lrs)
    ):
        raise ValueError("minimum learning rate must be finite and between zero and lr")

    def schedule(base_lr):
        def scale(epoch):
            if warmup_epochs and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            ratio = min_lr / base_lr
            return ratio + (1 - ratio) * 0.5 * (
                1 + math.cos(math.pi * min(progress, 1.0))
            )

        return scale

    return LambdaLR(optimizer, [schedule(lr) for lr in base_lrs])
