# Review against the official implementation

This audit compares the educational repository with `jchengai/pluto` at
`b9964b649c660f1f4a971d614c66f5992e24c18a`, the revision pinned in `AGENTS.md`.
It covers the guide, model/configuration, feature normalization, supervision,
optimizer/scheduler, augmentation and inference boundaries. The local nuPlan
adapters and simulator remain approximations. This review does not establish
checkpoint compatibility or reproduction of the paper's results.

## Findings and changes

| Finding | Change | Verification |
| --- | --- | --- |
| Every parameter received AdamW weight decay, unlike the released trainer. | Shared `recipes.make_optimizer` uses deterministic, sorted groups. Biases, normalization/embedding weights and free learned parameters have zero decay. Eligible linear/convolution/attention/recurrent weights decay. Training, benchmark, tuner and notebook use this helper. | Parameter coverage/uniqueness, representative module classes and an analytical zero-gradient update test; optimizer moments and groups match across interrupted/uninterrupted training. |
| The local cosine schedule approached zero, whereas the official trainer sets `min_lr=1e-6`. | Recipes explicitly record the minimum rate. Warmup/cosine uses the released formula and stays at the floor after the configured horizon. The zero-warmup smoke path remains supported. | Analytical rates at warmup, cosine and floor boundaries, separate group rates, invalid settings and checkpoint/scheduler resume. |
| The normalized-feature converter required future observations even when used for inference. | Explicit `include_targets=False` reads history only and returns no supervision keys. Training retains its future-data requirement. | History-only inputs with no future projection produce the same model inputs as a full example; finite inference, input immutability and rejection of missing training futures. |
| Target frames and projected-mode labels had only an all-valid fixture. | Added varied-heading, variable-horizon, padded-reference, zero-reference and empty-supervision cases. Correct existing coordinate and mode formulas were retained. | Agent position/heading targets are relative to that agent's current pose while vector axes remain the current ego frame; padded references cannot win labels; empty supervision yields zero loss and finite gradients. |
| History dimensions were documented in the wrong order. | Corrected the channel description to width then length, matching the official builder. | Checked against the pinned builder; tensor arithmetic is unchanged. |

The official [optimizer setup](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/src/models/pluto/pluto_trainer.py#L400-L492)
defines the decay policy and minimum rate; its
[scheduler](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/src/optim/warmup_cos_lr.py#L35-L57)
defines the epoch formula. The local schedule clamps after its horizon; the
released expression is periodic if stepped beyond training. Local optimizer
groups omit frozen parameters and reject unclassified weight-bearing modules.
These are deliberate engineering choices.

The [feature normalizer](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/src/features/pluto_feature.py#L174-L246),
[reference projections](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/src/feature_builders/pluto_feature_builder.py#L671-L716)
and [planning labels](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/src/models/pluto/pluto_trainer.py#L198-L230)
support the retained frame and label semantics. The guide recommends
[cache and training sanity checks](https://github.com/jchengai/pluto/blob/b9964b649c660f1f4a971d614c66f5992e24c18a/README.md#L42-L96)
before a full run. No full training, dataset download or checkpoint download
was started for this audit.

## Verification

The full offline CPU suite passes **102 tests**, including notebook/package
output and gradient parity, learning, FP32/BF16 resume, tracking and the new
regressions. Ruff lint/format, notebook source/asset integrity, deterministic
regeneration and fresh-process execution also pass. Both notebook editions were
regenerated; the twelve illustrations and their GitHub PNG paths are unchanged.

An additional bounded comparison extracted the pinned upstream
`configure_optimizers` method and `WarmupCosLR.get_lr` expression without
importing nuPlan/Lightning or the upstream model. On the local smoke model,
the upstream and local optimizer groups contain exactly the same parameter
identities in the same sorted order (74 decay, 124 no-decay). All 26 scheduler
positions from epoch 0 through 25 agree within `2.2e-19` absolute learning rate.
The previous local scheduler was also checked and returned zero at its final
position. This is optimizer/schedule evidence on the local model, not complete
model, cached-feature or training parity.

Re-run the supported local checks from the repository root with the documented
environment:

```sh
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q -p no:cacheprovider tests
ruff check pluto scripts tests notebooks/*.ipynb
ruff format --check pluto scripts tests
python scripts/check_notebook.py
python scripts/run_notebook.py
```

## Remaining gaps

1. **Features and data:** verify conversion on actual official caches and compare
   complete nuPlan feature construction, routes, actor selection and rasters.
   The local raw-data path is a bounded heuristic diagnostic.
2. **Architecture and initialization:** the local history attention, Fourier
   layout, query positional encoding, trajectory heads, pooling and PyTorch
   default initialization differ from the released implementation. Matching
   major dimensions does not imply operator or weight compatibility.
3. **Objectives and causal views:** SDF normalization/boundary policies and
   reference-free-head channels differ. Synthetic contrastive views do not
   establish causal validity on nuPlan scenes.
4. **Planning and evaluation:** the educational tracker/rules are not the
   official PDM/nuPlan evaluator. Released checkpoints, exact train/validation
   splits, full training, closed-loop scores and CUDA precision/memory remain
   unverified.

CPU tests and local Codex source review cover this change. No independent hosted
model review or complete shared-playbook CI compliance is claimed. Delivery of
the current educational implementation follows the task-scoped authorization
recorded in [the paper brief](paper-brief.md); remaining scientific gates stay
open. See [the paper-to-code map](paper-to-code.md) for the full deviation table.
