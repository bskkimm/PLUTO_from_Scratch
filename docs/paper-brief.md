# Paper brief

- Paper: Cheng et al., PLUTO, arXiv:2404.14327v1 (2024), https://arxiv.org/abs/2404.14327v1.
- Official code: https://github.com/jchengai/pluto/tree/b9964b649c660f1f4a971d614c66f5992e24c18a .
- Playbook: `707c64b138d5e6c534af3d197fd4f278aab03d45`; remote main verified 2026-11-16 through GitHub connector. Direct Git DNS unavailable.
- Audience: developers learning vector-based planning and PyTorch. Target: educational core; this version is incomplete, not official-equivalent.
- Data: seeded synthetic inputs only. Paper used nuPlan 1M frames and Val14 closed-loop scenarios. Data paths, official split files, weights and simulator are not supplied.
- Weights: random initialization; no pretrained weights loaded.
- Hardware observed: CPU; CUDA unavailable and NVIDIA driver inaccessible. No budget or authorization for full training was given.
- Paper recipe: batch 128 on four RTX 3090, 25 epochs, AdamW 1e-3, weight decay 1e-4, three-epoch warmup then cosine. This repository's synthetic smoke settings are different.
- Evaluation: synthetic average displacement error is a debugging metric only; real reproduction requires official nuPlan reactive/non-reactive closed-loop evaluation.
- Review/CI: no project Git remote or independent reviewer configured. Hosted CI, protection, and PR are unmet gates.
