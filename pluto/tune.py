"""Opt-in bounded CPU thread tuning for the unchanged smoke training recipe."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
from .benchmark import benchmark
from .losses import imitation_loss
from .model import Pluto, PlutoConfig
from .synthetic import sample_batch


def _fixture():
    torch.manual_seed(928)
    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
        state_dropout=0.0,
    )
    model = Pluto(config)
    batch = sample_batch(2, 4, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    predictions, gradients, losses = None, None, []
    for step in range(3):
        optimizer.zero_grad()
        output = model(batch)
        loss = imitation_loss(output, batch)["total"]
        loss.backward()
        if step == 0:
            predictions = output["trajectory"].detach().clone()
            gradients = [
                p.grad.detach().clone() if p.grad is not None else None
                for p in model.parameters()
            ]
        losses.append(loss.detach().clone())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return predictions, gradients, torch.stack(losses)


def _same(left, right):
    return (
        torch.allclose(left[0], right[0], rtol=1e-5, atol=1e-6)
        and torch.allclose(left[2], right[2], rtol=1e-5, atol=1e-6)
        and all(
            (a is None and b is None)
            or (
                a is not None
                and b is not None
                and torch.allclose(a, b, rtol=1e-5, atol=1e-6)
            )
            for a, b in zip(left[1], right[1])
        )
    )


def tune(output, threads=(1, 2), iterations=5, warmup=2, repeats=2, max_seconds=30.0):
    """Try CPU thread counts, accepting only numerically consistent measurable gains.

    The total deadline is checked between fixed-size trials, so one trial can
    finish after it. Architecture, synthetic data, batch=2, objective, FP32,
    optimizer and fixed epoch learning rate are unchanged. Not a GPU or microbatch tuner.
    """
    if (
        not threads
        or len(threads) > 8
        or len(set(threads)) != len(threads)
        or any(t < 1 or t > 16 for t in threads)
    ):
        raise ValueError("provide 1–8 unique thread counts between 1 and 16")
    if (
        not 1 <= iterations <= 100
        or not 0 <= warmup <= 20
        or not 2 <= repeats <= 5
        or not 0 < max_seconds <= 120
    ):
        raise ValueError("tuner bounds exceeded")
    started = time.monotonic()
    previous = torch.get_num_threads()
    trials = []
    result = {
        "effective_batch_size": 2,
        "fixture": "synthetic smoke FP32",
        "tuner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "iterations": iterations,
        "warmup_steps": warmup,
        "repeats": repeats,
        "max_seconds": max_seconds,
        "deadline_scope": "checked between trials",
        "trials": trials,
        "selected_threads": threads[0],
        "selection": "baseline unless correctness and >=5% median end-to-end gain are measured",
    }
    baseline, baseline_rate = None, None
    try:
        for index, count in enumerate(threads):
            row = {"threads": count, "accepted": False}
            trials.append(row)
            if index and time.monotonic() - started >= max_seconds:
                row["reason"] = "time budget exhausted before trial"
                continue
            try:
                torch.set_num_threads(count)
                fixture = _fixture()
                correct = baseline is None or _same(baseline, fixture)
                if baseline is None:
                    baseline = fixture
                if not correct:
                    row["reason"] = "FP32 output, gradient or learning parity failed"
                    continue
                measurements = [
                    benchmark(iterations, warmup, 2) for _ in range(repeats)
                ]
                runs = [
                    measurement["paths"]["end_to_end"]["samples_per_second"]
                    for measurement in measurements
                ]
                if "environment" not in result:
                    result["environment"] = {
                        key: measurements[0][key]
                        for key in (
                            "python",
                            "torch",
                            "cpu_model",
                            "code_base_revision",
                            "code_dirty",
                            "source_sha256",
                            "dataset",
                            "device",
                        )
                    }
                row["peak_process_rss_kib_linux"] = max(
                    m["peak_process_rss_kib_linux"] for m in measurements
                )
                rate = statistics.median(runs)
                row.update(
                    {
                        "samples_per_second": runs,
                        "median_samples_per_second": rate,
                        "accepted": True,
                    }
                )
                if index == 0:
                    baseline_rate = rate
                    row["reason"] = "baseline"
                elif (
                    baseline_rate is not None
                    and rate > baseline_rate * 1.05
                    and rate
                    > next(
                        t["median_samples_per_second"]
                        for t in trials
                        if t["threads"] == result["selected_threads"]
                    )
                ):
                    result["selected_threads"] = count
                    row["reason"] = "correct and >5% faster than baseline"
                else:
                    row["reason"] = "correct but no proven gain over selected baseline"
            except Exception as exc:
                row["reason"] = f"trial failed: {type(exc).__name__}: {exc}"
                if index == 0:
                    raise RuntimeError(
                        "baseline trial failed; no tuning result can be selected"
                    ) from exc
    finally:
        torch.set_num_threads(previous)
        result["elapsed_seconds"] = time.monotonic() - started
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    print(
        json.dumps(
            tune(
                args.output,
                tuple(args.threads),
                args.iterations,
                args.warmup,
                args.repeats,
                args.max_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
