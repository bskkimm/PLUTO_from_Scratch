"""Bounded CPU throughput measurements on the synthetic smoke fixture."""

import argparse
import hashlib
from pathlib import Path
import json
import platform
import resource
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader
from .losses import imitation_loss
from .model import Pluto
from .recipes import get_recipe, make_optimizer
from .synthetic import sample_batch


class SyntheticSamples(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {k: v.squeeze(0) for k, v in sample_batch(1, 4, seed=index).items()}


def benchmark(iterations=10, warmup=2, batch_size=2):
    if iterations < 1 or warmup < 0 or batch_size < 1:
        raise ValueError(
            "iterations and batch size must be positive; warmup nonnegative"
        )
    torch.manual_seed(123)
    init = time.perf_counter()
    loader = DataLoader(
        SyntheticSamples((iterations + warmup) * batch_size),
        batch_size=batch_size,
        num_workers=0,
    )
    cached = next(iter(loader))
    recipe = get_recipe("smoke")
    model = Pluto(recipe.model)
    optimizer = make_optimizer(model, recipe.learning_rate, recipe.weight_decay)
    startup = time.perf_counter() - init

    def update(batch):
        optimizer.zero_grad()
        loss = imitation_loss(model(batch), batch)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        return float(loss.detach())

    for _ in range(warmup):
        update(cached)
    rows = {}
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(iterations):
        next(iterator)
    elapsed = time.perf_counter() - start
    rows["loader"] = {
        "seconds": elapsed,
        "samples_per_second": iterations * batch_size / elapsed,
    }
    start = time.perf_counter()
    for _ in range(iterations):
        loss = update(cached)
    elapsed = time.perf_counter() - start
    rows["cached_step"] = {
        "seconds": elapsed,
        "samples_per_second": iterations * batch_size / elapsed,
        "last_loss": loss,
    }
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    wait = 0.0
    start = time.perf_counter()
    for _ in range(iterations):
        before = time.perf_counter()
        batch = next(iterator)
        wait += time.perf_counter() - before
        loss = update(batch)
    elapsed = time.perf_counter() - start
    rows["end_to_end"] = {
        "seconds": elapsed,
        "loader_wait_seconds": wait,
        "samples_per_second": iterations * batch_size / elapsed,
        "last_loss": loss,
    }
    source_root = Path(__file__).resolve().parent.parent
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(git_root).resolve() != source_root:
            raise ValueError("source directory is not the Git root")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=source_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        revision, dirty = None, None
    cpu_model = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in open("/proc/cpuinfo")
                if line.startswith("model name")
            ),
            "unavailable",
        )
        if platform.system() == "Linux"
        else platform.processor()
    )
    return {
        "device": "cpu",
        "cpu_model": cpu_model,
        "code_base_revision": revision,
        "code_dirty": dirty,
        "source_sha256": {
            f"pluto/{name}": hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "model.py",
                "losses.py",
                "synthetic.py",
                "recipes.py",
                "benchmark.py",
            )
        },
        "dataset": "generated synthetic samples; no disk cache",
        "workers": 0,
        "intraop_threads": torch.get_num_threads(),
        "iterations": iterations,
        "warmup_steps": warmup,
        "batch_size": batch_size,
        "startup_seconds": startup,
        "peak_process_rss_kib_linux": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "paths": rows,
        "scope": "CPU wall time; cached and end-to-end include loss, backward, gradient clipping and optimizer step at fixed epoch learning rate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(benchmark(args.iterations, args.warmup, args.batch_size), indent=2)
    )


if __name__ == "__main__":
    main()
