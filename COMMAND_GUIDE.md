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
```

For prepared data, use `--manifest path/to/train.json` on training and `--manifest path/to/validation.json` on evaluation. Each manifest's JSON contains `format: "pluto-batches-v1"`, `split`, and `batches: [{"path": "batch.pt", "sha256": "..."}]`. Batch paths must stay under the manifest directory and contain tensor dictionaries with the keys in `pluto/synthetic.py`. Hashes are verified before loading. Set the recipe to match the time horizon. Do not use the same split as evidence of generalization.

MLflow is optional; `python -m pip install '.[tracking]'` installs it when authorized. `mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` opens the local UI. Checkpoints are trusted local files loaded with Python pickle; never load an untrusted checkpoint. Exact resume is tested only for single-process synthetic CPU and a matching environment. Manifest CIL, worker-process recovery, real-data and GPU recovery are unverified.
