"""Build the offline notebook from the actual local module definitions."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "notebooks/assets/architecture"
ASSETS.mkdir(parents=True, exist_ok=True)


def overview_svg():
    """Editable pipeline figure with explicit inference and supervision paths."""
    boxes = [
        (25, 82, 210, 68, "SCENE INPUT", "agents · map · static", "#d6e8f7"),
        (275, 82, 220, 68, "SCENE ENCODER", "masked tokens [B,N,D]", "#dcefe4"),
        (865, 82, 235, 68, "AGENT FORECAST", "one future per agent", "#f9e9ce"),
        (25, 235, 210, 68, "REFERENCE LINES", "valid polylines [B,R,P]", "#d6e8f7"),
        (275, 235, 220, 68, "QUERY GRID", "R lines × M learned modes", "#dcefe4"),
        (550, 235, 230, 68, "FACTORIZED DECODER", "lateral → speed → scene", "#dcefe4"),
        (865, 235, 235, 68, "PROPOSALS + SCORES", "[B,R,M,T,6] and [B,R,M]", "#f9e9ce"),
        (865, 340, 235, 68, "SELECT / FALLBACK", "rule scores or free head", "#f9e9ce"),
        (25, 420, 210, 68, "TRAINING TARGETS", "future · masks · SDF", "#f5dce0"),
        (550, 420, 230, 68, "TRAINING OBJECTIVE", "imitation + aux + CIL", "#f5dce0"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="535" viewBox="0 0 1140 535">',
        '<rect width="100%" height="100%" fill="#f5f8fc"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#465d76"/></marker></defs>',
        '<text x="25" y="43" font-family="sans-serif" font-size="25" font-weight="bold" fill="#18263d">PLUTO: vector scene → candidate plans → selection</text>',
        '<text x="25" y="517" font-family="sans-serif" font-size="15" fill="#465d76">Solid arrows: forward data flow · dashed arrow: training gradient · D=128, M=12, T=80 in the paper</text>',
    ]
    for x, y, width, height, heading, detail, fill in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="9" fill="{fill}" stroke="#465d76"/>'
        )
        parts.append(
            f'<text x="{x + 12}" y="{y + 26}" font-family="sans-serif" font-weight="bold" font-size="15" fill="#18263d">{escape(heading)}</text>'
        )
        parts.append(
            f'<text x="{x + 12}" y="{y + 50}" font-family="sans-serif" font-size="14" fill="#18263d">{escape(detail)}</text>'
        )
    for path in [
        "M235 116 H275",
        "M495 116 H865",
        "M235 269 H275",
        "M495 269 H550",
        "M385 150 V188 H665 V235",
        "M780 269 H865",
        "M980 303 V340",
        "M235 454 H550",
        "M1100 116 H1120 V454 H780",
        "M1100 269 H1120",
        "M495 116 H805 V374 H865",
    ]:
        parts.append(
            f'<path d="{path}" fill="none" stroke="#465d76" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    parts.append(
        '<text x="788" y="360" font-family="sans-serif" font-size="13" fill="#465d76">free head</text>'
    )
    parts.append(
        '<path d="M550 440 H520 V303 H550" fill="none" stroke="#a54e5a" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arrow)"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def worked_figure(fid):
    """Mechanism-specific editable SVGs; numeric examples are illustrative."""
    if fid not in {
        "data-flow",
        "history-pyramid",
        "query-grid",
        "loss",
        "shapes",
        "contrastive",
        "inference",
        "bicycle",
    }:
        return None
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="410" viewBox="0 0 1140 410">',
        '<rect width="100%" height="100%" fill="#f5f8fc"/>',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#465d76"/></marker></defs>',
    ]

    def text(x, y, label, size=17, bold=False, color="#18263d"):
        weight = "bold" if bold else "normal"
        parts.append(
            f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(label)}</text>'
        )

    def box(x, y, w, h, color="#dcefe4"):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{color}" stroke="#465d76"/>'
        )

    def arrow(path, dashed=False):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="#465d76" stroke-width="2"{dash} marker-end="url(#a)"/>'
        )

    if fid == "data-flow":
        text(25, 38, "Data contract: history pairs, masks and supervision", 24, True)
        text(25, 78, "Toy history: 5 observations (1 = observed)", 17, True)
        for i, value in enumerate((1, 1, 1, 0, 1)):
            box(35 + i * 73, 95, 55, 43, "#d6e8f7" if value else "#e5e8ec")
            text(54 + i * 73, 123, str(value), 18, True)
        text(35, 167, "t0", 14)
        text(108, 167, "t1", 14)
        text(181, 167, "t2", 14)
        text(254, 167, "t3", 14)
        text(327, 167, "t4", 14)
        arrow("M410 116 H460")
        text(470, 89, "Adjacent-pair mask", 17, True)
        for i, value in enumerate((1, 1, 0, 0)):
            box(475 + i * 73, 100, 55, 43, "#dcefe4" if value else "#e5e8ec")
            text(494 + i * 73, 128, str(value), 18, True)
        text(
            25,
            218,
            "9 features per pair: Δx, Δy, Δvx, Δvy, cos(Δθ), sin(Δθ), width, length, valid",
            16,
        )
        box(25, 250, 315, 80, "#d6e8f7")
        text(40, 279, "Map: [B,P,N,10]", 16, True)
        text(40, 305, "centerline + left/right boundaries", 15)
        box(390, 250, 315, 80, "#dcefe4")
        text(405, 279, "References: [B,R,N,6]", 16, True)
        text(405, 305, "masked points → lateral queries", 15)
        box(755, 250, 345, 80, "#f5dce0")
        text(770, 279, "Future: [B,T,6]", 16, True)
        text(770, 305, "x, y, cos θ, sin θ, vx, vy", 15)
        text(
            25,
            375,
            "Frames: current ego-centered · distances: m · velocity: m/s · masks: True means valid",
            15,
            color="#465d76",
        )
    elif fid == "history-pyramid":
        text(25, 38, "Masked temporal attention pyramid", 24, True)
        text(
            25,
            72,
            "Paper-sized example: 20 history vectors; D=128. Local attention is not NATTEN-verified.",
            16,
        )
        specs = [
            (95, "20 × 32", "local window 3"),
            (190, "10 × 64", "local window 3"),
            (285, "5 × 128", "local window 5"),
        ]
        for y, shape, desc in specs:
            box(40, y, 210, 60, "#d6e8f7")
            text(55, y + 25, shape, 17, True)
            text(55, y + 47, desc, 14)
            box(385, y, 220, 60, "#dcefe4")
            text(400, y + 25, "Project to 128", 16, True)
            text(400, y + 47, "temporal lateral convolution", 14)
            arrow(f"M250 {y + 30} H385")
        arrow("M145 155 V190")
        arrow("M145 250 V285")
        arrow("M605 315 H650 V220 H605")
        arrow("M605 220 H650 V125 H605")
        box(740, 95, 340, 60, "#f9e9ce")
        text(755, 120, "Fuse + read last valid", 16, True)
        text(755, 142, "agent token [B,A,128]", 14)
        arrow("M605 125 H740")
        text(
            40,
            380,
            "Mask padded inputs → zero padded outputs; resampling preserves finite empty-agent output.",
            15,
            color="#465d76",
        )
    elif fid == "query-grid":
        text(25, 38, "Factorized lateral–longitudinal attention", 24, True)
        text(25, 75, "Toy example: B=1, R=2 references, M=3 learned modes, D=16", 16)
        for i in range(2):
            text(48, 153 + i * 74, f"line {i}", 16, True)
            for j in range(3):
                box(150 + j * 96, 120 + i * 74, 72, 50, "#dcefe4")
                text(168 + j * 96, 151 + i * 74, f"q{i}{j}", 17, True)
        for j in range(3):
            text(155 + j * 96, 94, f"mode {j}", 14)
        arrow("M185 100 V116")
        arrow("M281 100 V116")
        arrow("M377 100 V116")
        arrow("M440 170 H500")
        box(510, 100, 245, 62, "#d6e8f7")
        text(525, 126, "1 · lateral attention", 16, True)
        text(525, 147, "across reference rows", 14)
        box(510, 190, 245, 62, "#dcefe4")
        text(525, 216, "2 · mode attention", 16, True)
        text(525, 237, "across mode columns", 14)
        arrow("M632 162 V190")
        box(830, 145, 270, 62, "#f9e9ce")
        text(845, 171, "3 · scene cross-attention", 16, True)
        text(845, 192, "queries read scene tokens", 14)
        arrow("M755 220 H790 V176 H830")
        text(
            25,
            326,
            "Masked references cannot be keys or selected outputs. Complexity uses R²M + RM², not (RM)².",
            15,
            color="#465d76",
        )
    elif fid == "loss":
        text(
            25,
            38,
            "Training objective: chosen plan, distance field and triplets",
            24,
            True,
        )
        box(25, 75, 315, 85, "#d6e8f7")
        text(40, 104, "Imitation + prediction", 17, True)
        text(40, 130, "Smooth L1 + mode cross-entropy", 15)
        box(395, 75, 315, 85, "#f5dce0")
        text(410, 104, "Distance-field penalty", 17, True)
        text(410, 130, "max(0, radius + margin − d)", 15)
        box(765, 75, 335, 85, "#dcefe4")
        text(780, 104, "Contrastive objective", 17, True)
        text(780, 130, "original ↔ positive vs negative", 15)
        text(30, 212, "Illustrative SDF example", 18, True)
        text(30, 242, "radius = 1.0 m, margin = 0.2 m, sampled distance = 0.7 m", 16)
        box(30, 260, 420, 55, "#f9e9ce")
        text(45, 294, "hinge = max(0, 1.0 + 0.2 − 0.7) = 0.5 m", 16, True)
        text(520, 218, "Positive + original: supervised", 16, True)
        text(520, 249, "Negative: scene encoder + contrastive loss only", 16)
        text(520, 280, "SDF lookup is bilinear and differentiable", 16)
        text(
            25,
            375,
            "Teacher-forced candidate labels are inputs; displayed numbers are illustrative, not measured results.",
            15,
            color="#465d76",
        )
    elif fid == "shapes":
        text(25, 38, "One complete toy forward pass", 24, True)
        text(25, 75, "B=2, R=2, M=3, T=4, D=16; full paper dimensions differ", 16)
        box(25, 110, 230, 64, "#d6e8f7")
        text(40, 136, "Scene [2,N,16]", 17, True)
        text(40, 157, "agent/map/static tokens", 14)
        box(25, 205, 230, 64, "#d6e8f7")
        text(40, 231, "References [2,2,6,6]", 17, True)
        text(40, 252, "and modes [3,16]", 14)
        box(345, 205, 230, 64, "#dcefe4")
        text(360, 231, "Query [2,2,3,16]", 17, True)
        text(360, 252, "R × M combinations", 14)
        arrow("M255 237 H345")
        box(660, 135, 245, 105, "#f9e9ce")
        text(675, 162, "Trajectories", 17, True)
        text(675, 185, "[2,2,3,4,6]", 17)
        text(675, 216, "logits [2,2,3]", 15)
        arrow("M255 142 H625 V160 H660")
        arrow("M575 237 H625 V210 H660")
        box(940, 145, 175, 85, "#f5dce0")
        text(955, 173, "Selected", 17, True)
        text(955, 198, "[2,4,6]", 16)
        arrow("M905 187 H940")
        text(
            25,
            344,
            "A missing reference set uses the separately decoded free trajectory; padded references cannot win.",
            15,
            color="#465d76",
        )
    elif fid == "contrastive":
        text(25, 38, "Contrastive views: same encoder, different supervision", 24, True)
        labels = [
            (90, "original", "#d6e8f7"),
            (180, "positive", "#dcefe4"),
            (270, "negative", "#f5dce0"),
        ]
        for y, label, color in labels:
            box(25, y, 170, 55, color)
            text(40, y + 34, label, 17, True)
            arrow(f"M195 {y + 27} H280")
        box(280, 90, 225, 235, "#dcefe4")
        text(295, 138, "Shared scene", 18, True)
        text(295, 164, "encode all views", 16)
        text(295, 205, "project only for triplet", 15, True)
        box(660, 110, 395, 72, "#f9e9ce")
        text(675, 139, "Triplet contrastive loss", 17, True)
        text(675, 162, "compare sim(z,z⁺) with sim(z,z⁻)", 15)
        box(660, 235, 395, 72, "#f5dce0")
        text(675, 264, "Supervised objective", 17, True)
        text(675, 287, "decode original + positive only", 15)
        arrow("M505 160 H660")
        arrow("M505 270 H660")
        text(
            25,
            375,
            "Negative targets are not used. Whether a view preserves or changes causality must be established by preprocessing.",
            15,
            color="#465d76",
        )
    elif fid == "inference":
        text(25, 38, "Synthetic inference and open-loop ADE", 24, True)
        text(
            25,
            78,
            "Toy logits: [2.0, 1.0] → choose candidate 0; padded references are excluded.",
            16,
        )
        box(25, 110, 300, 65, "#d6e8f7")
        text(40, 138, "Candidate 0", 17, True)
        text(40, 160, "(0,0), (2,0) metres", 15)
        box(25, 205, 300, 65, "#f5dce0")
        text(40, 233, "Target", 17, True)
        text(40, 255, "(0,0), (1,0) metres", 15)
        arrow("M325 142 H430")
        arrow("M325 237 H430")
        box(430, 130, 300, 120, "#dcefe4")
        text(445, 161, "Point distances", 17, True)
        text(445, 190, "0 m and 1 m", 16)
        text(445, 222, "mean = 0.5 m", 18, True)
        arrow("M730 190 H815")
        box(815, 145, 290, 85, "#f9e9ce")
        text(830, 175, "Synthetic ADE", 17, True)
        text(830, 200, "0.5 m (illustrative)", 15)
        text(
            25,
            345,
            "The official benchmark uses closed-loop simulation; this open-loop diagnostic is not its score.",
            15,
            color="#465d76",
        )
    elif fid == "bicycle":
        text(25, 38, "Kinematic bicycle and finite-horizon tracking", 24, True)
        box(35, 100, 330, 120, "#d6e8f7")
        text(50, 130, "State: x, y, yaw, speed", 17, True)
        text(50, 158, "Control: acceleration, steering", 15)
        text(50, 185, "dt = 0.1 s; distance in m", 15)
        arrow("M365 160 H430")
        box(430, 100, 320, 120, "#dcefe4")
        text(445, 130, "x' = x + v cos(yaw) dt", 16, True)
        text(445, 158, "y' = y + v sin(yaw) dt", 16)
        text(445, 185, "yaw' = yaw + v tan(δ) dt / L", 15)
        arrow("M750 160 H815")
        box(815, 100, 290, 120, "#f9e9ce")
        text(830, 130, "Straight toy example", 17, True)
        text(830, 158, "x=0, v=1 m/s, yaw=0", 15)
        text(830, 185, "after one step: x=0.1 m", 15)
        box(35, 270, 1070, 60, "#f5dce0")
        text(
            50,
            296,
            "Reference states → backward LQR gains → forward bicycle rollout → external rule scorer",
            16,
            True,
        )
        text(
            25,
            375,
            "Illustrative tracker; not yet validated against the official PDM/nuPlan simulator.",
            15,
            color="#465d76",
        )
    parts.append("</svg>")
    return "".join(parts)


def figure(fid, title, rows):
    content = overview_svg() if fid == "overview" else worked_figure(fid)
    if content is not None:
        (ASSETS / f"{fid}.svg").write_text(content)
        return content
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
    "toy_sdf = torch.full((1, 5, 5), 0.7)\ntoy_trajectory = torch.tensor([[[0., 0., 1., 0.]]], requires_grad=True)\ntoy_penalty = sdf_loss(toy_trajectory, toy_sdf, resolution=1., offsets=(0.,), radius=1., margin=0.2)\nassert torch.allclose(toy_penalty, torch.tensor(0.5), atol=1e-6)\nprint('illustrative hinge (m):', toy_penalty.item())\n",
    "sdf-probe",
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
            "pluto/simulation.py",
            "pluto/synthetic.py",
            "scripts/make_notebook.py",
            "scripts/render_figures.py",
        ]
    },
    "generation": "python scripts/make_notebook.py",
    "inspection": "See inspection.md for individual visual inspection and remaining limits.",
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(ASSETS / "README.md").write_text(
    "# Figure inventory\n\nTen editable SVG illustrations are embedded as notebook attachments. The manifest identifies their cell anchors and hashes. They include illustrative values and schematic flows. See inspection.md for inspection evidence and limits. Regenerate vector attachments with `python scripts/make_notebook.py` and previews with `python scripts/render_figures.py`.\n"
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
