# PLUTO from Scratch

An educational PyTorch implementation of [PLUTO](https://arxiv.org/abs/2404.14327v1), with a runnable CPU example and an illustrated teaching notebook.

The implementation is compared with the [official PLUTO code at revision b9964b6](https://github.com/jchengai/pluto/tree/b9964b649c660f1f4a971d614c66f5992e24c18a).

The implementation includes scene encoding, factorized trajectory decoding, imitation and contrastive losses, training checkpoints, and diagnostic evaluation. It remains incomplete: official nuPlan reproduction, pretrained-weight compatibility, full training, and GPU validation are unverified. Synthetic results are diagnostics.

## Quick start

Use Python 3.10 or 3.11. The package pins PyTorch 2.9.1.

```sh
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1
python -m pluto.train --out runs/smoke --epochs 3
python -m pluto.evaluate runs/smoke/checkpoint.pt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests
```

The evaluation command uses the same seeded synthetic batch as training. Its
displacement error is a smoke-test diagnostic, not a measure of generalization
or nuPlan planning performance. Use a new output directory for each run, or add
`--resume` to continue a compatible checkpoint.

## Notebook and documentation

- [Illustrated walkthrough](notebooks/pluto_e2e_walkthrough.ipynb) — model definitions, worked examples, and 12 figures.
- [Self-contained offline notebook](notebooks/pluto_e2e_walkthrough_offline.ipynb) — the same lesson with embedded images.
- [Command guide](COMMAND_GUIDE.md) — training, evaluation, data preparation, and optional features.
- [Paper-to-code mapping](docs/paper-to-code.md) — implementation details and deviations.
- [Implementation review](docs/implementation-review.md) — source audit, fixes, and remaining gaps.
- [Paper brief](docs/paper-brief.md) — sources, validation scope, and remaining work.
