# Paper brief

- Paper: Cheng et al., PLUTO, arXiv:2404.14327v1 (2024), https://arxiv.org/abs/2404.14327v1.
- Official code: https://github.com/jchengai/pluto/tree/b9964b649c660f1f4a971d614c66f5992e24c18a .
- Playbook: `707c64b138d5e6c534af3d197fd4f278aab03d45`; remote main verified 2026-11-16 through GitHub connector. Direct Git DNS unavailable.
- Audience: developers learning vector-based planning and PyTorch. Target: complete educational paper implementation; current work remains incomplete and not official-equivalent.
- Data: nuPlan mini logs and maps are present under `/home/beomseokkim2/dataset/nuplan`; one read-only ego-only window was probed (see `reports/nuplan-ego-diagnostic.json`). The paper used nuPlan 1M frames and Val14 closed-loop scenarios; those exact split identities and official weights have not been established. A separate local devkit checkout exists but its import failed because `geopandas` is absent from the tested environment.
- Weights: random initialization; no pretrained weights loaded.
- Hardware observed: CPU; CUDA unavailable and NVIDIA driver inaccessible. No budget or authorization for full training was given.
- Paper recipe: batch 128 on four RTX 3090, 25 epochs, AdamW 1e-3, weight decay 1e-4, three-epoch warmup then cosine. This repository's synthetic smoke settings are different.
- Evaluation: synthetic average displacement error is a debugging metric only; real reproduction requires official nuPlan reactive/non-reactive closed-loop evaluation.
- Delivery: user authorized committing, pushing, reviewing and merging this project after the required gates. This intentionally overrides the playbook's generic no-merge default. Private repository and draft PR #1 exist; basic hosted CPU CI passed at the earlier head, but full CI, current-head trusted review, and branch protection are unmet gates.
