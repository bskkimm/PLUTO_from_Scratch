"""Opt-in, read-only single-window nuPlan ego-log/model diagnostic."""

import argparse
from dataclasses import asdict
import platform
import subprocess
import hashlib
import json
from pathlib import Path
import torch
from pluto.model import Pluto, PlutoConfig
from pluto.nuplan_ego import read_ego_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--timestamp",
        type=int,
        help="center timestamp in microseconds; defaults near log start",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    batch = read_ego_only(args.database, center_timestamp=args.timestamp)
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
    digest = hashlib.sha256()
    with args.database.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    result = {
        "code_base_revision": revision,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_config": asdict(config),
        "database_basename": args.database.name,
        "database_sha256": digest.hexdigest(),
        "center_timestamp_requested_us": args.timestamp,
        "history_samples": 21,
        "future_samples": 80,
        "seed": 123,
        "device": "cpu",
        "weights": "random",
        "features": "ego only; no map, agents or reference lines",
        "selected_shape": list(output["selected"].shape),
        "finite_output": bool(torch.isfinite(output["selected"]).all()),
        "limitation": "read-only integration diagnostic; not training, official nuPlan evaluation, or planning-quality evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
