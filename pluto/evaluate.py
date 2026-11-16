"""Open-loop displacement diagnostic; not an official nuPlan benchmark."""

import argparse
import json
import torch
from .dataset import BatchManifest
from .model import Pluto, PlutoConfig
from .synthetic import sample_batch


def evaluate_checkpoint(checkpoint, manifest=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = PlutoConfig(**state["config"]["model"])
    data = BatchManifest(manifest) if manifest is not None else None
    if data is None and state["config"]["dataset"] != "synthetic-v1":
        raise ValueError(
            "provide an explicit evaluation manifest for a manifest-trained model"
        )
    model = Pluto(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    batches = (
        (data[i] for i in range(len(data)))
        if data
        else (sample_batch(2, cfg.future_steps, seed=state["config"]["seed"]),)
    )
    total, count, examples = 0.0, 0, 0
    with torch.no_grad():
        for batch in batches:
            selected = model(batch)["selected"]
            valid = batch["target_mask"]
            distance = (selected[..., :2] - batch["target"][..., :2]).norm(dim=-1)
            total += (distance * valid).sum().item()
            count += valid.sum().item()
            examples += valid.shape[0]
    if count == 0:
        raise ValueError("evaluation has no valid target points")
    return {
        "ade_m": total / count,
        "examples": examples,
        "valid_points": count,
        "dataset": {"manifest_sha256": data.identity, "split": data.split}
        if data
        else "synthetic-v1",
        "metric": "open-loop selected ADE; not nuPlan closed-loop score",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", help="local preprocessed evaluation batches")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.manifest)))


if __name__ == "__main__":
    main()
