"""Opt-in, read-only single-window nuPlan ego-log/model diagnostic."""

import argparse
from dataclasses import asdict
import platform
import subprocess
import sqlite3
import hashlib
import json
from pathlib import Path
import torch
from pluto.model import Pluto, PlutoConfig
from pluto.nuplan_ego import read_ego_only
from pluto.nuplan_actors import read_ego_and_actors
from pluto.nuplan_map import add_map_features, read_route_context
from pluto.nuplan_ego import _yaw
from pluto.losses import imitation_loss


def diagnostic_step(model, batch):
    """Run exactly one opt-in CPU step; results are not training metrics."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = imitation_loss(model(batch), batch)["total"]
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite diagnostic loss")
    loss.backward()
    finite = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not finite:
        raise RuntimeError("non-finite diagnostic gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
    optimizer.step()
    return {
        "steps": 1,
        "finite_loss": True,
        "finite_gradients": finite,
        "loss": loss.item(),
        "gradient_norm": grad_norm,
        "meaning": "randomly initialized single-batch diagnostic, not a planning metric",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--timestamp",
        type=int,
        help="center timestamp in microseconds; defaults near log start",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--optimization-step",
        action="store_true",
        help="run one CPU loss/backward/update diagnostic",
    )
    parser.add_argument(
        "--map", type=Path, help="nuPlan GeoPackage; requires --timestamp"
    )
    parser.add_argument(
        "--actors",
        action="store_true",
        help="include tracked actors; requires --timestamp",
    )
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    if (args.actors or args.map) and args.timestamp is None:
        parser.error("--actors and --map require an explicit --timestamp")
    batch = (
        read_ego_and_actors(args.database, center_timestamp=args.timestamp)
        if args.actors
        else read_ego_only(args.database, center_timestamp=args.timestamp)
    )
    if args.map:
        with sqlite3.connect(
            args.database.resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            ego = connection.execute(
                "SELECT x,y,qw,qx,qy,qz,epsg FROM ego_pose WHERE timestamp<=? ORDER BY timestamp DESC LIMIT 1",
                (args.timestamp,),
            ).fetchone()
        route, traffic = read_route_context(args.database, args.timestamp)
        batch = add_map_features(
            batch,
            args.map,
            (ego[0], ego[1], _yaw(*ego[2:6])),
            int(ego[6]),
            modes=3,
            route_ids=route,
            traffic_states=traffic,
        )
    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=80,
        dropout=0.0,
    )
    model = Pluto(config).eval()
    with torch.no_grad():
        output = model(batch)
    step_result = diagnostic_step(model, batch) if args.optimization_step else None
    digest = hashlib.sha256()
    with args.database.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    map_sha = None
    if args.map:
        map_digest = hashlib.sha256()
        with args.map.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                map_digest.update(block)
        map_sha = map_digest.hexdigest()
    if args.map:
        import pyproj
        import shapely

        map_libraries = {"pyproj": pyproj.__version__, "shapely": shapely.__version__}
    else:
        map_libraries = None
    result = {
        "code_base_revision": revision,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_config": asdict(config),
        "map_libraries": map_libraries,
        "source_sha256": {
            str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (
                Path("pluto/model.py"),
                Path("pluto/data.py"),
                Path("pluto/losses.py"),
                Path("pluto/nuplan_ego.py"),
                Path("pluto/nuplan_actors.py"),
                Path("pluto/nuplan_map.py"),
                Path("scripts/probe_nuplan_ego.py"),
            )
        },
        "database_basename": args.database.name,
        "database_sha256": digest.hexdigest(),
        "center_timestamp_requested_us": args.timestamp,
        "history_samples": 21,
        "future_samples": 80,
        "seed": 123,
        "device": "cpu",
        "weights": "random",
        "features": "ego, optional actors, and heuristically connected route references"
        if args.map
        else (
            "ego and tracked actors; no map or reference lines"
            if args.actors
            else "ego only; no map, agents or reference lines"
        ),
        "map_basename": args.map.name if args.map else None,
        "map_version_directory": args.map.parent.name if args.map else None,
        "map_sha256": map_sha,
        "map_count": batch["map"].shape[1],
        "reference_count": batch["reference"].shape[1],
        "reference_lengths_m": batch["reference"][0, :, :, 2:4]
        .norm(dim=-1)
        .sum(dim=-1)
        .tolist(),
        "on_route_map_count": int(batch["on_route"].sum()) if args.map else None,
        "actor_count": batch["agents"].shape[1] - 1,
        "static_count": batch["static"].shape[1],
        "selected_shape": list(output["selected"].shape),
        "candidate_shape": list(output["trajectory"].shape)
        if output["trajectory"] is not None
        else None,
        "finite_output": bool(torch.isfinite(output["selected"]).all()),
        "optimization_step": step_result,
        "limitation": "read-only data integration diagnostic with optional one-step update; reference paths use heuristic lane connectivity, not the official reference builder; not full training, official nuPlan evaluation, or planning-quality evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
