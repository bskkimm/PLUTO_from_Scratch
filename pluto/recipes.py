"""Named model-size and optimizer settings; see fidelity notes in docs."""

from dataclasses import dataclass
import math
from torch.optim.lr_scheduler import LambdaLR
from .model import PlutoConfig


@dataclass(frozen=True)
class Recipe:
    model: PlutoConfig
    learning_rate: float
    weight_decay: float
    warmup_epochs: int


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


def make_scheduler(optimizer, epochs, warmup_epochs):
    if epochs < 1 or warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup must be nonnegative and shorter than total epochs")

    def scale(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, scale)
