# Paper to code and deviations

Sources: [paper v1](https://arxiv.org/html/2404.14327v1) and [official code at b9964b6](https://github.com/jchengai/pluto/tree/b9964b649c660f1f4a971d614c66f5992e24c18a). This repository is a partial educational implementation, not a checkpoint-compatible port.

| Paper | Local implementation | Evidence and limitations |
| --- | --- | --- |
| III-B scene representation | `PointEncoder`, `Pluto.scene_encoder` | Agent/map/static tokens and masks. Simplified max-pooled agent histories and map features replace the released specialized history, state-attention, Fourier position, category and map encoders. Features are synthetic and not official-feature compatible. |
| III-C Eq. 4–6 | `Pluto`, `DecoderLayer` | Reference and learned longitudinal embeddings form a Cartesian query grid; self-attention across each axis then scene cross-attention. Separate official head layout and positional embeddings differ. |
| III-C Eq. 7–11 | `free_head`, `imitation_loss`, `prediction_head` | Six channels `[x,y,cos,sin,vx,vy]`; losses sum over channels per valid timestep. Caller supplies mode labels; `assign_modes` provides an elementary polyline-projection helper, not yet verified against official projection/preprocessing. Paper describes a free head, but released configuration makes it optional and uses four channels when enabled; local head has six. |
| III-D Eq. 12 | `sdf_loss` | Differentiable circle-center lookup and hinge cost, averaged per time point per equation. Paper pseudocode/released collision implementation normalize by active circles and use different boundary masks. Local center and out-of-bounds policy differ; no cost-map construction or integration in trainer. |
| III-E Eq. 13 | `contrastive_loss` | Implements normalized positive-vs-negative softmax for externally supplied embeddings. Causal augmentations and the 3-way training pipeline are absent. |
| III-F | `selected` | Selects highest learned logit, with a free fallback if references are missing. Does not perform top-k filtering, LQR/bicycle rollouts, rule scores, collision checks or closed-loop evaluation. |
| IV-B | `PlutoConfig`; `pluto.train` | Config defaults follow major published dimensions, but not official operator details. Training CLI uses a small synthetic smoke recipe with cosine schedule, no warmup and no augmentation. It is not the paper recipe. |

`True` means valid throughout. Padded values must be finite; at least one ego token is always valid. Reference padding cannot be selected. Position channels represent ego-frame metres; heading is a cosine/sine pair; velocity is metres per second. Agent prediction targets are agent-relative. Real preprocessing, exact tensor semantics, map construction, official scoring, and compatibility remain unverified.
