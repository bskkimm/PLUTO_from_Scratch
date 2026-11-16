# Commands

Use Python 3.11, PyTorch 2.9.1, and pytest 9.1.1. Local validation used `/home/beomseokkim2/miniconda3/envs/torch_env/bin/python`, `OMP_NUM_THREADS=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `PYTHONPATH=.`. The system Python provides an incompatible old PyTorch; no packages were installed.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests
OMP_NUM_THREADS=1 python -m pluto.train --out runs/demo --epochs 3 --stop-after 1
OMP_NUM_THREADS=1 python -m pluto.train --out runs/demo --epochs 3 --resume
OMP_NUM_THREADS=1 python -m pluto.evaluate runs/demo/checkpoint.pt
```

Checkpoints are trusted local files loaded with Python pickle; do not load untrusted checkpoints. Exact recovery is tested only for the single-process synthetic CPU fixture and matching software/precision. Real data, worker processes and GPU recovery are not supported or tested.
