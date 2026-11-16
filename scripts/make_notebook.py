"""Build the offline notebook from the actual local module definitions."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "notebooks/assets/architecture"
ASSETS.mkdir(parents=True, exist_ok=True)


def figure(fid, title, rows):
    width, height = 1100, 110 + 86 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f8fc"/>',
        f'<text x="30" y="45" font-family="sans-serif" font-size="25" font-weight="bold" fill="#18263d">{escape(title)}</text>',
    ]
    for i, row in enumerate(rows):
        y = 75 + i * 86
        for j, item in enumerate(row):
            x = 30 + j * 350
            parts += [
                f'<rect x="{x}" y="{y}" width="300" height="58" rx="9" fill="{["#d6e8f7", "#dcefe4", "#f9e9ce"][j % 3]}" stroke="#465d76"/>',
                f'<text x="{x + 12}" y="{y + 34}" font-family="sans-serif" font-size="17" fill="#18263d">{escape(item)}</text>',
            ]
            if j < len(row) - 1:
                parts.append(
                    f'<path d="M{x + 302} {y + 29} h 38 m -8 -6 l 8 6 -8 6" fill="none" stroke="#465d76" stroke-width="2"/>'
                )
    parts.append("</svg>")
    text = "".join(parts)
    (ASSETS / f"{fid}.svg").write_text(text)
    return text


def md(text, id, fig=None, title="", rows=()):
    cell = {
        "cell_type": "markdown",
        "id": id,
        "metadata": {},
        "source": text.splitlines(True),
    }
    if fig:
        cell["source"] += [f"\n![{title}](attachment:{fig}.svg)\n"]
        cell["attachments"] = {
            f"{fig}.svg": {"image/svg+xml": figure(fig, title, rows)}
        }
    cells.append(cell)


def code(source, id):
    cells.append(
        {
            "cell_type": "code",
            "id": id,
            "metadata": {},
            "source": source.splitlines(True),
            "execution_count": None,
            "outputs": [],
        }
    )


cells = []
md(
    "# PLUTO: from vector scene to trajectory\n\nThis offline lesson implements a **simplified educational variant**, with randomly initialized weights and toy synthetic data. It does not reproduce nuPlan or the official code. See `docs/paper-to-code.md` in the repository.\n",
    "intro",
    "overview",
    "Overview: synthetic scene to selected plan",
    [
        ("Agent / map / static", "Masked scene encoder", "Scene tokens"),
        ("Reference polylines", "Lateral × speed queries", "Trajectory + scores"),
        ("Targets / SDF / triplets", "Training losses", "Optimizer update"),
    ],
)
code(
    "import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom dataclasses import dataclass\ntorch.manual_seed(7)\ntorch.set_num_threads(1)\n",
    "setup",
)
md(
    "## Data contract\n\nMasks use `True` for valid points. The first agent is ego; positions are metres in the ego frame. This synthetic fixture supplies mode labels directly; an elementary projection helper is available but is not connected to real preprocessing. Shapes below use batch size 2 and a four-step horizon.\n",
    "data",
    "data-flow",
    "Input masks and labels",
    [
        ("Agent histories [B,A,H,9]", "Temporal attention", "Agent tokens [B,A,D]"),
        (
            "Map [B,P,N,10]; refs [...,6]",
            "Mask padded points",
            "Polyline tokens [B,R,D]",
        ),
    ],
)
synthetic = (ROOT / "pluto/synthetic.py").read_text().replace("import torch\n", "")
code(
    synthetic
    + "\nbatch = sample_batch(batch_size=2, future_steps=4)\nprint({k: tuple(v.shape) for k, v in batch.items()})\n",
    "synthetic",
)
md(
    "## Masked history pyramid\n\nThe history encoder applies local self-attention at three temporal scales, combines the scales with lateral projections, and reads the last valid original timestep. The windows and mask behavior are tested, but exact NATTEN operator parity is not established.\n",
    "history",
    "history-pyramid",
    "Masked temporal attention pyramid",
    [
        ("History length T", "Local windows of 3", "Downsample by 2"),
        ("Second / third scales", "Project and upsample", "Last valid timestep"),
    ],
)
md(
    "## Architecture\n\nMap point embeddings are pooled over valid elements; map points include left/right lane boundaries and semantic attributes. Ego scalar features use attention with optional dropout. Queries combine one reference-line token and one learned longitudinal mode. Decoder layers attend across references, then modes, then encoded scene tokens. Empty references use the free head.\n",
    "arch",
    "query-grid",
    "Factorized query attention",
    [
        ("Reference R=2", "Cross with M=3 modes", "Query grid [B,2,3,D]"),
        ("Attend across R", "Attend across M", "Cross-attend scene"),
    ],
)
model_source = (ROOT / "pluto/model.py").read_text()
for statement in (
    "from dataclasses import dataclass\n",
    "import math\n",
    "import torch\n",
    "from torch import nn\n",
    "import torch.nn.functional as F\n",
):
    model_source = model_source.replace(statement, "")
code(model_source, "model-definitions")
md(
    "## Assemble and inspect\n\nThe toy network uses D=16 and one layer per stack, unlike the paper configuration (D=128, four layers and 12 longitudinal queries).\n",
    "assembly",
    "shapes",
    "Toy shape trace",
    [
        ("Scene [B,N,16]", "2 references × 3 modes", "6 trajectory candidates"),
        ("Candidate [4,6]", "Score each candidate", "Select best or free"),
    ],
)
code(
    "cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4, dropout=0.0)\nmodel = Pluto(cfg)\nout = model(batch)\nprint('plans', out['trajectory'].shape, 'other agents', out['prediction'].shape)\n",
    "forward",
)
md(
    "## Objectives and optimization\n\nThe toy imitation loss uses a supplied reference/mode label, Smooth L1 trajectory and prediction terms and classification. The paper also trains with collision/off-road and contrastive terms; the standalone utilities below are not wired into this toy training step. SDF lookup is differentiable bilinear interpolation.\n",
    "objective",
    "loss",
    "Training signals and gradient flow",
    [
        ("Chosen query + target", "Smooth L1 and CE", "Backprop to model"),
        ("Circle centers + SDF", "Bilinear lookup + hinge", "Optional spatial loss"),
        ("Anchor / positive / negative", "Cosine softmax", "Optional contrastive loss"),
    ],
)
loss_source = (
    (ROOT / "pluto/losses.py")
    .read_text()
    .replace("import torch\n", "")
    .replace("import torch.nn.functional as F\n", "")
)
code(loss_source, "loss-definitions")
code(
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)\nlosses = imitation_loss(out, batch)\noptimizer.zero_grad()\nlosses['total'].backward()\noptimizer.step()\nprint({k: round(v.item(), 4) for k, v in losses.items()})\n",
    "update",
)
md(
    "## Contrastive imitation\n\nA valid positive view preserves the target; a negative view changes a relevant cause, so its old target is not used. All three are encoded, but only original and positive views are decoded for supervision. The example is synthetic and its causal meaning is not validated.\n",
    "cil",
    "contrastive",
    "Contrastive views and supervision",
    [
        ("Original + positive", "Imitation + auxiliary", "Supervised loss"),
        (
            "Original / positive / negative",
            "Shared scene encoder",
            "Triplet contrastive loss",
        ),
    ],
)
objective_source = (
    (ROOT / "pluto/objective.py")
    .read_text()
    .replace("from .losses import contrastive_loss, imitation_loss, sdf_loss\n", "")
    .replace("import torch\n", "")
)
code(objective_source, "objective-definitions")
code(
    "positive = {k: v.clone() for k, v in batch.items()}\nnegative = {k: v.clone() for k, v in batch.items()}\nnegative['agent_mask'][:, -1] = False\ncil_loss = contrastive_objective(model, batch, positive, negative)\noptimizer.zero_grad()\ncil_loss['total'].backward()\noptimizer.step()\nprint({k: round(v.item(), 4) for k, v in cil_loss.items()})\n",
    "cil-step",
)
md(
    "## Inference and evaluation\n\nLearned logits select a candidate. The paper additionally uses simulation-based scoring, which is absent here. This average displacement error (ADE) is a synthetic open-loop diagnostic, not a nuPlan score.\n",
    "evaluation",
    "inference",
    "Synthetic inference",
    [
        ("Predicted logits", "Choose highest score", "Selected [B,T,6]"),
        ("Compare xy in metres", "Mean point distance", "Synthetic ADE"),
    ],
)
code(
    "model.eval()\nwith torch.no_grad():\n    pred = model(batch)['selected']\n    ade = (pred[..., :2] - batch['target'][..., :2]).norm(dim=-1).mean()\nprint('synthetic ADE (m):', float(ade))\n",
    "metric",
)
md(
    "## Checkpoint workflow\n\nThe separate `pluto.train` module implements an epoch-boundary synthetic checkpoint containing model, optimizer, scheduler, RNG state and config. Exact resume is only tested for CPU single-process synthetic data. Optional local MLflow tracking is implemented; raw nuPlan integration and full hosted CI remain future work.\n",
    "checkpoint",
    "checkpoint",
    "Epoch-boundary recovery",
    [
        ("Model + optimizer + RNG", "Write temporary file", "Atomic replace"),
        ("Resume matching config", "Restore epoch and state", "Continue training"),
    ],
)
manifest = {
    "figures": [
        {
            "id": cell["id"],
            "attachment": list(cell["attachments"])[0],
            "sha256": hashlib.sha256(
                next(iter(cell["attachments"].values()))["image/svg+xml"].encode()
            ).hexdigest(),
        }
        for cell in cells
        if "attachments" in cell
    ],
    "sources": {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [
            "pluto/model.py",
            "pluto/losses.py",
            "pluto/objective.py",
            "pluto/synthetic.py",
            "scripts/make_notebook.py",
        ]
    },
    "generation": "python scripts/make_notebook.py",
    "inspection": "Automated generation only; individual visual inspection pending.",
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(ASSETS / "README.md").write_text(
    "# Figure inventory\n\nNine editable SVG illustrations are embedded as notebook attachments. The manifest identifies their cell anchors and hashes. They are schematic, not a substitute for pending detailed, visually inspected worked figures. Regenerate with `python scripts/make_notebook.py`.\n"
)
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "notebooks/pluto_e2e_walkthrough.ipynb").write_text(
    json.dumps(nb, indent=1) + "\n"
)
