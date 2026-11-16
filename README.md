# PLUTO from Scratch

An educational, **partial** plain-PyTorch implementation of the planning architecture in [PLUTO](https://arxiv.org/abs/2404.14327v1). It runs offline on synthetic CPU inputs. It is not a reproduction of the paper or an implementation ready for autonomous driving.

The model encodes vectorized agents, map polylines and static objects; combines reference-line embeddings with learned longitudinal modes; applies lateral, longitudinal and scene cross-attention; and predicts trajectories, scores and other-agent futures. It also has a reference-free fallback. Loss utilities cover imitation, prediction, a contrastive triplet objective and differentiable signed-distance-field penalties. See [paper-to-code](docs/paper-to-code.md) for important limitations and source differences.

## Quick start

Python 3.11 and PyTorch 2.9.1 were used for local CPU verification. In an environment with those packages and pytest installed:

```sh
export OMP_NUM_THREADS=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests
python -m pluto.train --out runs/smoke --epochs 3
python -m pluto.evaluate runs/smoke/checkpoint.pt
```

The training example is deliberately tiny and synthetic. Use a new output directory for a new run; `--resume` resumes an existing checkpoint at an epoch boundary with identical configuration. `--stop-after 1` simulates an interruption while retaining the original total epochs. The progress estimate uses only the current session, including checkpoint time, and is provisional after its first epoch.

The [walkthrough](notebooks/pluto_e2e_walkthrough.ipynb) defines the model and objectives locally and runs with synthetic inputs; stored outputs are empty. The synthetic schema is shown in `pluto/synthetic.py`; its features are illustrative, not a nuPlan feature builder. `True` means valid in every mask. Positions use ego-frame metres, and agent prediction targets are agent-relative. The caller must supply a valid `mode_label` (reference index times number of modes plus longitudinal index); `pluto.losses.assign_modes` provides a segment-projection helper but is not yet connected to a real-data pipeline.

## Status and boundaries

- **Implemented:** masked vector encoding, factorized query attention, trajectory and agent heads, fallback, imitation and prediction objectives, separate SDF and triplet loss utilities, synthetic train/eval and epoch checkpoint resume.
- **Not implemented:** nuPlan loading and preprocessing, nuPlan-specific target generation, the paper's CIL augmentation pipeline, cost-map generation, integrating SDF/CIL into training, simulator/LQR/rule scoring, official metrics, MLflow, full benchmark/tuner, production-quality illustrated notebook verification and complete CI/review workflow.
- **Not verified:** parity with official operators/checkpoints, GPU precision, real dataset execution, or closed-loop performance. Random weights are used. No downloaded data or expensive training was run.
- **Reference recipe only:** 128 hidden channels, 4 encoder/decoder layers, 4 heads, 12 modes, 80 future steps; the small synthetic smoke recipe changes model size and horizon and is not a faithful training recipe.

See [commands](COMMAND_GUIDE.md), [paper brief](docs/paper-brief.md), and [mapping and deviations](docs/paper-to-code.md). The paper's benchmark claims remain the authors' results; this repository has no reproduced metrics.
