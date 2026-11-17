"""Opt-in preparation of a single local nuPlan diagnostic batch and manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import torch
from .nuplan_actors import read_ego_and_actors
from .nuplan_ego import _yaw
from .nuplan_map import add_map_features, read_route_context


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_batch_manifest(batch, output_dir, split, provenance):
    """Publish each file atomically without clobbering; manifest is published last."""
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    if not isinstance(batch, dict) or not all(
        isinstance(v, torch.Tensor) for v in batch.values()
    ):
        raise ValueError("batch must contain tensors only")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch_path, manifest_path = output / "batch.pt", output / "manifest.json"
    if batch_path.exists() or manifest_path.exists():
        raise FileExistsError("prepared batch or manifest already exists")
    temp_path = None
    manifest_temp = None
    created_batch = False
    created_manifest = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".batch-", delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            torch.save({k: v.detach().cpu() for k, v in batch.items()}, temporary)
        os.link(
            temp_path, batch_path
        )  # Atomic create that refuses to replace an existing file.
        created_batch = True
        manifest = {
            "format": "pluto-batches-v1",
            "split": split,
            "batches": [{"path": batch_path.name, "sha256": _file_sha256(batch_path)}],
            "provenance": provenance,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output, prefix=".manifest-", delete=False
        ) as target:
            manifest_temp = Path(target.name)
            target.write(payload)
        os.link(manifest_temp, manifest_path)
        created_manifest = True
    except Exception:
        if created_manifest:
            manifest_path.unlink(missing_ok=True)
        if created_batch:
            batch_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)
    return manifest_path


def prepare_nuplan(log_path, map_path, timestamp, output_dir, split, modes=12):
    """Prepare one heuristic nuPlan window; not official feature-builder parity."""
    log_path, map_path = (
        Path(log_path).resolve(strict=True),
        Path(map_path).resolve(strict=True),
    )
    if modes < 2:
        raise ValueError("modes must be at least 2")
    batch = read_ego_and_actors(log_path, timestamp)
    with sqlite3.connect(log_path.as_uri() + "?mode=ro", uri=True) as connection:
        ego = connection.execute(
            "SELECT x,y,qw,qx,qy,qz,epsg FROM ego_pose WHERE timestamp<=? ORDER BY timestamp DESC LIMIT 1",
            (timestamp,),
        ).fetchone()
    route, lights = read_route_context(log_path, timestamp)
    batch = add_map_features(
        batch,
        map_path,
        (ego[0], ego[1], _yaw(*ego[2:6])),
        int(ego[6]),
        route_ids=route,
        traffic_states=lights,
        modes=modes,
    )
    provenance = {
        "kind": "nuplan-heuristic-single-window-v1",
        "log_basename": log_path.name,
        "log_sha256": _file_sha256(log_path),
        "map_version": map_path.parent.name,
        "map_sha256": _file_sha256(map_path),
        "timestamp_us": timestamp,
        "modes": modes,
        "history_samples": 21,
        "future_samples": 80,
        "interval_us": 100000,
        "limitations": "heuristic reference paths; no official feature-builder parity",
    }
    return write_batch_manifest(batch, output_dir, split, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument(
        "--timestamp",
        type=int,
        required=True,
        help="planning timestamp in microseconds",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split", required=True, help="honest split identifier, e.g. mini-diagnostic"
    )
    parser.add_argument("--modes", type=int, default=12)
    args = parser.parse_args()
    print(
        prepare_nuplan(
            args.log, args.map, args.timestamp, args.out, args.split, args.modes
        )
    )


if __name__ == "__main__":
    main()
