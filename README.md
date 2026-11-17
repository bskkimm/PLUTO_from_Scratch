# PLUTO from Scratch

An educational, **incomplete** plain-PyTorch implementation of [PLUTO](https://arxiv.org/abs/2404.14327v1). It runs offline on synthetic CPU inputs and can consume explicitly prepared tensor batches. It is not an official-equivalent reproduction or a system ready for autonomous driving.

The planner encodes vectorized agents, map polylines and static objects with position features; combines reference-line embeddings with learned longitudinal modes; applies lateral, longitudinal and scene cross-attention; and predicts trajectories, scores and other-agent futures. It includes a reference-free fallback, imitation and prediction losses, a contrastive triplet objective, and a differentiable signed-distance-field penalty. See [paper-to-code](docs/paper-to-code.md) for material limitations.

## Quick start

Python 3.11 and PyTorch 2.9.1 were used for local CPU verification. With those packages and pytest installed:

```sh
export OMP_NUM_THREADS=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests
python -m pluto.train --out runs/smoke --epochs 3
python -m pluto.evaluate runs/smoke/checkpoint.pt
python -m pluto.benchmark --iterations 5 --warmup 2
```

For a toy contrastive example, add `--cil` to the training command. Its synthetic perturbations are diagnostic and are not validated causal augmentations. With a prepared manifest, `--cil` requires externally validated `interactive` and `leading` boolean agent masks; optional traffic substitutions and SDF tensors enable their corresponding losses. Optional MLflow tracking uses `--mlflow-uri sqlite:///runs/mlflow.db` and requires the `tracking` extra. Use a new output directory for each run; `--resume` resumes an existing checkpoint at an epoch boundary with identical configuration. The current-session progress estimate includes checkpoint time and is provisional after the first epoch; total active time across resumes is approximate. Device and precision options include CPU FP32/BF16 and CUDA FP32/BF16/FP16; CUDA remains unverified.

The [walkthrough](notebooks/pluto_e2e_walkthrough.ipynb) defines the model and objectives locally, defaults to seeded synthetic CPU execution, and has no stored outputs. `pluto/synthetic.py` documents the model's tensor contract. Masks use `True` for valid values; positions are ego-frame metres and agent future positions are agent-relative. Callers must supply valid reference/mode labels. `pluto.official_adapter.convert_normalized_batch` can convert already-normalized, batched official-style features; raw nuPlan preprocessing is not included or verified.

## Prepared batches and recipes

A `pluto-batches-v1` manifest contains `split` and a nonempty `batches` list with relative `path` and `sha256` for each tensor dictionary. The loader checks all hashes and rejects path escapes; training records the manifest digest and rejects a changed manifest on resume. Pass `--manifest path/to/manifest.json` to train, and explicitly supply a manifest to evaluate a manifest-trained checkpoint. The current batch schema and recipe horizon must match. This is an offline tensor interface, not an official dataset adapter.

`--recipe smoke` is the default. `practical` and `paper_shape` provide larger model shapes and an epoch-based three-epoch warmup/cosine schedule. `paper_shape` uses the paper's major dimensions (128 channels, 4 encoder/decoder layers, 12 modes, 80 future steps), but does not make the implementation or data pipeline faithful. Full training requires a suitable manifest, compute and explicit authorization.

## Remaining work

Full nuPlan scene extraction, official history/map encoder parity, complete causal augmentation and automatically generated real-data cost maps, official LQR/bicycle simulation and rule scoring, official weights/operator parity, GPU checks, GPU and data-loader tuning, full CI/review and illustrated-notebook gates remain unfinished. The candidate selector accepts an external evaluator; it does not implement those safety rules. Synthetic ADE and [CPU timings](reports/cpu-smoke-benchmark.json) are diagnostics, not nuPlan results. A single read-only ego-only window from the local nuPlan mini dataset was probed with random weights; no complete real-scene evaluation, pretrained weights, GPU evaluation, or full training was used.

The opt-in CPU tuner compares a bounded set of thread counts on the synthetic FP32 smoke recipe. It checks output/gradient/learning parity, measures repeated end-to-end steps, and retains the baseline unless the tested candidate is at least 5% faster; it does not tune GPU or microbatch settings. The GitHub Actions workflow does not yet satisfy the full playbook contract. See [commands](COMMAND_GUIDE.md), [paper brief](docs/paper-brief.md), and [mapping and deviations](docs/paper-to-code.md).
