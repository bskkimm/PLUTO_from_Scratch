"""Small deterministic synthetic training example; not the nuPlan recipe."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from itertools import islice
import os
from pathlib import Path
import time
import platform
import uuid
import torch
from .model import Pluto
from .recipes import get_recipe, make_scheduler
from .dataset import BatchManifest
from .losses import imitation_loss
from .augmentation import make_causal_views, mask_agents, perturb_state
from .objective import contrastive_objective
from .progress import estimate_progress
from .synthetic import sample_batch
from .tracking import Tracker


def run_training(
    out_dir,
    epochs=3,
    seed=123,
    resume=False,
    stop_after=None,
    cil=False,
    manifest=None,
    tracking_uri=None,
    recipe="smoke",
    accumulation_steps=1,
):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_recipe(recipe)
    cfg = settings.model
    data = BatchManifest(manifest) if manifest is not None else None
    dataset_id = (
        {"manifest_sha256": data.identity, "split": data.split}
        if data
        else "synthetic-v1"
    )
    config = {
        "model": asdict(cfg),
        "epochs": epochs,
        "seed": seed,
        "dataset": dataset_id,
        "batch_size": 2 if data is None else "from_manifest",
        "lr": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "warmup_epochs": settings.warmup_epochs,
        "recipe": recipe,
        "accumulation_steps": accumulation_steps,
        "cil": cil,
        "tracking_uri": tracking_uri,
    }
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists() and not resume:
        raise FileExistsError(
            "checkpoint exists; use --resume or a new output directory"
        )
    torch.manual_seed(seed)
    model = Pluto(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = make_scheduler(optimizer, epochs, settings.warmup_epochs)
    start = 0
    run_id = None
    first_started = datetime.now().astimezone()
    previous_active = 0.0
    optimizer_steps = 0
    if resume:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["config"] != config:
            raise ValueError("resume config differs from checkpoint config")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng"])
        start = ckpt["epoch"]
        run_id = ckpt.get("run_id")
        first_started = datetime.fromisoformat(ckpt["first_started"])
        previous_active = ckpt["active_before_checkpoint"]
        optimizer_steps = ckpt["optimizer_steps"]
    tracker = Tracker(tracking_uri, config, run_id) if tracking_uri else None
    run_id = tracker.run_id if tracker else None
    session_dir = out_dir / "sessions"
    session_dir.mkdir(exist_ok=True)
    session = {
        "config": config,
        "resume_epoch": start,
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (session_dir / (uuid.uuid4().hex + ".json")).write_text(
        json.dumps(session, indent=2) + "\n"
    )
    synthetic_batch = (
        sample_batch(batch_size=2, future_steps=cfg.future_steps, seed=seed)
        if data is None
        else None
    )
    start_time = time.monotonic()
    limit = min(epochs, stop_after if stop_after is not None else epochs)
    for epoch in range(start, limit):
        model.train()
        epoch_losses = []
        component_totals = {}
        batches = (
            (data[i] for i in range(len(data)))
            if data is not None
            else (synthetic_batch,)
        )
        iterator = iter(batches)
        gradient_norms = []
        while True:
            group = list(islice(iterator, accumulation_steps))
            if not group:
                break
            group_examples = sum(batch["target"].shape[0] for batch in group)
            if not group_examples:
                raise ValueError("empty accumulation group")
            optimizer.zero_grad()
            for batch in group:
                if batch["target"].shape[1] != cfg.future_steps:
                    raise ValueError(
                        "batch target horizon differs from the chosen recipe"
                    )
                if cil:
                    if data is None:
                        positive = perturb_state(
                            batch, torch.randn_like(batch["ego_state"]) * 0.05
                        )
                        drop = torch.zeros_like(batch["agent_mask"][..., 0])
                        drop[:, -1] = True
                        negative = mask_agents(batch, drop)
                        contrast_valid = (
                            None  # Toy views, not validated causal nuPlan augmentation.
                        )
                    else:
                        if "interactive" not in batch or "leading" not in batch:
                            raise ValueError(
                                "manifest CIL requires interactive and leading masks"
                            )
                        positive, negative, contrast_valid = make_causal_views(
                            batch,
                            batch["interactive"],
                            batch["leading"],
                            batch.get("traffic_mask"),
                            batch.get("traffic_replacement"),
                        )
                    components = contrastive_objective(
                        model,
                        batch,
                        positive,
                        negative,
                        sdf=batch.get("sdf"),
                        contrast_valid=contrast_valid,
                    )
                else:
                    components = imitation_loss(model(batch), batch)
                loss = components["total"]
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                (loss * (batch["target"].shape[0] / group_examples)).backward()
                epoch_losses.append(loss.item())
                for name, value in components.items():
                    if name != "total":
                        component_totals[name] = (
                            component_totals.get(name, 0.0) + value.detach().item()
                        )
            gradient_norms.append(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            )
            optimizer.step()
            optimizer_steps += 1
        component_totals["gradient_norm"] = (
            sum(gradient_norms) / len(gradient_norms) * len(epoch_losses)
        )
        scheduler.step()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "config": config,
            "run_id": run_id,
            "first_started": first_started.isoformat(),
            "active_before_checkpoint": previous_active + time.monotonic() - start_time,
        }
        temporary = out_dir / "checkpoint.pt.tmp"
        torch.save(state, temporary)
        os.replace(temporary, ckpt_path)
        timing = estimate_progress(
            epoch + 1,
            epochs,
            epoch + 1 - start,
            time.monotonic() - start_time,
            previous_active,
            first_started,
            datetime.now().astimezone(),
        )
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if tracker:
            tracker.progress(
                epoch + 1,
                epochs,
                epoch_loss,
                optimizer.param_groups[0]["lr"],
                timing,
                {k: v / len(epoch_losses) for k, v in component_totals.items()},
            )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": epoch_loss,
                    **timing,
                }
            )
        )
    if tracker and limit == epochs and start < epochs:
        tracker.finish()
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--accumulate", type=int, default=1, help="microbatches per optimizer update"
    )
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--cil",
        action="store_true",
        help="enable diagnostic synthetic contrastive training",
    )
    parser.add_argument("--manifest", help="local preprocessed tensor-batch manifest")
    parser.add_argument(
        "--mlflow-uri",
        help="opt-in MLflow tracking URI (e.g. sqlite:///runs/mlflow.db)",
    )
    parser.add_argument(
        "--recipe", choices=["smoke", "practical", "paper_shape"], default="smoke"
    )
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("epochs and stop-after must be positive")
    run_training(
        args.out,
        args.epochs,
        args.seed,
        args.resume,
        args.stop_after,
        args.cil,
        args.manifest,
        args.mlflow_uri,
        args.recipe,
        args.accumulate,
    )


if __name__ == "__main__":
    main()
