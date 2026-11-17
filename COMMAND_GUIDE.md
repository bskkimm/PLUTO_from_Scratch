# Commands

Local checks used Python 3.11.14, PyTorch 2.9.1+cu130 in CPU mode, pytest 9.1.1, and MLflow 3.14.0. The system Python has an incompatible old PyTorch. No data or package download is necessary to run the commands in the prepared environment.

```sh
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q -p no:cacheprovider tests
python scripts/run_notebook.py
python -m pluto.train --out runs/demo --epochs 3 --stop-after 1
python -m pluto.train --out runs/demo --epochs 3 --resume
python -m pluto.evaluate runs/demo/checkpoint.pt
python -m pluto.train --out runs/cil --epochs 3 --cil
python -m pluto.train --out runs/tracked --epochs 3 --mlflow-uri sqlite:///runs/mlflow.db
python -m pluto.benchmark --iterations 5 --warmup 2
python scripts/probe_nuplan_ego.py /path/to/nuplan-log.db --output runs/ego-probe.json
python scripts/probe_nuplan_ego.py /path/to/nuplan-log.db --actors --timestamp MICROSECONDS --output runs/actor-probe.json
python scripts/probe_nuplan_ego.py /path/to/nuplan-log.db --actors --map /path/to/map.gpkg --timestamp MICROSECONDS --optimization-step --output runs/map-probe.json
python -m pluto.tune --output runs/tuning.json --threads 1 2 --iterations 4 --warmup 2 --repeats 2 --max-seconds 15
```

For prepared data, use `--manifest path/to/train.json` on training and `--manifest path/to/validation.json` on evaluation. Each manifest's JSON contains `format: "pluto-batches-v1"`, `split`, and `batches: [{"path": "batch.pt", "sha256": "..."}]`. Batch paths must stay under the manifest directory and contain tensor dictionaries with the keys in `pluto/synthetic.py`. Hashes are verified before loading. Set the recipe to match the time horizon. Do not use the same split as evidence of generalization.

MLflow is optional; `python -m pip install '.[tracking]'` installs it when authorized. `mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` opens the local UI. Checkpoints are trusted local files loaded with Python pickle; never load an untrusted checkpoint. Exact resume is tested only for single-process synthetic CPU and a matching environment. Manifest CIL accepts precomputed `interactive`/`leading` boolean masks of shape [B,A] (ego at index 0), optional `traffic_mask` and `traffic_replacement` tensors of shape [B,P], and optional `sdf` [B,H,W] in metres at 0.2 m/pixel. These inputs must be computed and causally validated by the caller. The CIL data path is tested on synthetic batches only. The tuner is CPU-only and compares its fixed effective batch of two. Worker-process recovery, real-data and GPU recovery are unverified.

`--accumulate N` groups up to N consecutive microbatches per optimizer update; the final group may be smaller and is weighted by sample count. Because losses average over valid labels and the model uses stochastic layers, this does not guarantee equivalence to a single concatenated batch. The learning-rate schedule advances per epoch. Progress uses the current session to estimate remaining work; saved total active time is approximate because work after the checkpoint snapshot is not included across restarts.

The optional nuPlan SQLite probe opens the database read-only, samples a single window, and can include tracked actors and nearby map features with route-filtered individual lane segments. Install the `nuplan-map` extra for map probing. Its optional one-step optimization uses random model weights; these probes are not official feature-builder parity, full training, closed-loop evaluation, or planning-quality evidence.

Training defaults to `--device cpu --precision fp32`. `--precision bf16` is available on CPU or supported CUDA devices, and `--precision fp16` requires CUDA; checkpoints include scaler state and CUDA random states. CPU BF16 resume is tested on the synthetic fixture; CUDA, real-data precision and GPU memory behavior are unverified. Resume requires the same device and precision configuration.

For `--build-sdf`, install the `science` extra and supply boolean `[B,H,W]` tensors named `drivable_mask` and `obstacle_mask` instead of `sdf` in each prepared batch. Distances use 0.2 m/pixel and cell-center Euclidean distance; rasterization and causal validity remain the caller’s responsibility.
