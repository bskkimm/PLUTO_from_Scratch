"""Build the offline notebook from the actual local module definitions."""

import hashlib
import json
from pathlib import Path
from notebook_figures import INVENTORY, draw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "notebooks/assets/architecture"
ASSETS.mkdir(parents=True, exist_ok=True)


def figure(fid, title, rows):
    content = draw(fid)
    (ASSETS / f"{fid}.svg").write_text(content)
    return content


def md(text, id, fig=None, title="", rows=()):
    cell = {
        "cell_type": "markdown",
        "id": id,
        "metadata": {},
        "source": text.splitlines(True),
    }
    if fig:
        _, question, _, caption = INVENTORY[fig]
        cell["source"] += [
            f"\n![{question} {caption}](attachment:{fig}.svg)\n",
            f"\n*Figure {list(INVENTORY).index(fig) + 1}. {caption}*\n",
        ]
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
    "## Scene tokens and geometric addresses\n\nA masked pointwise MLP and max pooling turn each polyline into a fixed-width token. An empty polyline gives zero. History, ego state, map and static-object tokens receive pose and type embeddings before scene attention. The waves below illustrate frequency encoding; their amplitudes are not model measurements.\n",
    "scene",
    "scene-tokens",
    "Polyline pooling, Fourier position and typed scene tokens",
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
code(
    "toy_points = torch.tensor([[[0.2,0.8,0.4], [0.9,0.1,0.3], [0.5,0.4,0.7]]])\ntoy_pool = PointEncoder(3, 3)\ntoy_pool.net = nn.Identity()\npooled = toy_pool(toy_points, torch.ones(1,3,dtype=torch.bool))\nassert torch.allclose(pooled, torch.tensor([[0.9,0.8,0.7]]))\nassert toy_pool(toy_points, torch.zeros(1,3,dtype=torch.bool)).eq(0).all()\nprint('illustrated pooled feature:', pooled.tolist())\n",
    "pooling-probe",
)
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
    "toy_sdf = torch.full((1, 5, 5), 0.7)\ntoy_trajectory = torch.tensor([[[0., 0., 1., 0.]]], requires_grad=True)\ntoy_penalty = sdf_loss(toy_trajectory, toy_sdf, resolution=1., offsets=(0.,), radius=1., margin=0.2)\nassert torch.allclose(toy_penalty, torch.tensor(0.5), atol=1e-6)\nprint('illustrative hinge (m):', toy_penalty.item())\n",
    "sdf-probe",
)
md(
    "## Follow the gradient into an update\n\nFor a scalar prediction 0 and target 1, Smooth L1 is 0.5 and its derivative with respect to the prediction is −1. The full model has many coupled parameters; AdamW uses moving averages and weight decay. The illustrated scalar direction explains the gradient, not the size of an AdamW parameter update.\n",
    "optimization",
    "optimizer",
    "Smooth L1 gradient and AdamW step order",
)
code(
    "toy_pred = torch.tensor([0.], requires_grad=True)\ntoy_loss = F.smooth_l1_loss(toy_pred, torch.tensor([1.]))\ntoy_gradient, = torch.autograd.grad(toy_loss, toy_pred)\nassert toy_loss.item() == 0.5 and toy_gradient.item() == -1.\nprint('toy Smooth L1:', toy_loss.item(), 'gradient:', toy_gradient.item())\n",
    "scalar-gradient",
)
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
    "## Inference and evaluation\n\nLearned logits select a candidate. The paper additionally uses an official simulation-based scoring system, which is not replicated here. This average displacement error (ADE) is a synthetic open-loop diagnostic, not a nuPlan score.\n",
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
    "## Bicycle rollout and tracking\n\nThe reference plan is tracked with a finite-horizon linear-quadratic controller and a kinematic bicycle. The 0.1-second, straight-line example is analytically checkable. This illustrative tracker has not been compared with the nuPlan/PDM implementation or official safety rules.\n",
    "simulation",
    "bicycle",
    "Straight reference and bicycle rollout",
    [
        ("Initial x=0 m, v=1 m/s", "Bicycle step dt=0.1 s", "Next x=0.1 m"),
        ("Future reference states", "Backward LQR gains", "Forward tracked states"),
    ],
)
simulation_source = (
    (ROOT / "pluto/simulation.py").read_text().replace("import torch\n", "")
)
code(simulation_source, "simulation-definitions")
code(
    "initial = torch.tensor([[0., 0., 0., 1.]])\nreference = torch.zeros(1, 1, 4, 6)\nreference[0, 0, :, 0] = torch.arange(1., 5.) * 0.1\nreference[..., 2] = 1.\nreference[..., 4] = 1.\nrollout = lqr_rollout(initial, reference)\nprint('tracked x (m):', rollout[0, 0, :, 0])\n",
    "simulation-probe",
)
md(
    "## Checkpoint workflow\n\nThe separate `pluto.train` module saves model, optimizer, scheduler, scaler, RNG state, epoch, configuration and implementation hashes at epoch boundaries. Resume requires matching configuration, manifest identity and implementation hashes. Exact resume is tested on CPU FP32/BF16 synthetic fixtures; CUDA and real-data resume remain unverified. Optional MLflow tracking and hosted CPU CI are implemented. The raw nuPlan adapter remains a heuristic one-window diagnostic.\n",
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
            "mime_type": "image/svg+xml",
            "question": INVENTORY[Path(list(cell["attachments"])[0]).stem][1],
            "caption": INVENTORY[Path(list(cell["attachments"])[0]).stem][3],
            "source_module": INVENTORY[Path(list(cell["attachments"])[0]).stem][2],
            "method": "Editable deterministic SVG; overview includes retained generated conceptual artwork",
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
            "pluto/simulation.py",
            "pluto/synthetic.py",
            "scripts/make_notebook.py",
            "scripts/notebook_figures.py",
            "scripts/render_figures.py",
            "pluto/data.py",
            "pluto/planner.py",
            "pluto/train.py",
            "notebooks/assets/illustrations/pluto-scene-to-plan.png",
            "notebooks/assets/illustrations/provenance.json",
        ]
    },
    "generation": "python scripts/make_notebook.py",
    "inspection": "See inspection.md for individual visual inspection and remaining limits.",
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(ASSETS / "README.md").write_text(
    "# Figure inventory\n\nExactly 12 distinct instructional figures are embedded as self-contained SVG attachments. The opening figure includes generated conceptual artwork; all computational layers, geometry, labels and dependencies remain editable in `scripts/notebook_figures.py`. The generated artwork and its exact prompt are retained in `../illustrations/`.\n\n"
    "| Figure | Cell anchor | Learning question | Source |\n| --- | --- | --- | --- |\n"
    + "\n".join(
        f"| {fid} | `{anchor}` | {question} | `{source}` |"
        for fid, (anchor, question, source, _) in INVENTORY.items()
    )
    + "\n\nEach figure follows its explanation and precedes the relevant probe or definition. The full model is defined locally in `model-definitions`; it follows the three encoder/decoder mechanism illustrations. Repeated attention blocks share one worked figure. The package's optional external rule evaluator appears beside learned selection; it is not implemented in the notebook.\n\n"
    "Regenerate with `python scripts/make_notebook.py` and `python scripts/render_figures.py`. Run `python scripts/check_notebook.py` and `python scripts/run_notebook.py`. Analytical arithmetic and model output/loss/gradient parity are checked in the tests. See `inspection.md` for visual inspection evidence and limits. Code outputs are deliberately empty in version control.\n"
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
