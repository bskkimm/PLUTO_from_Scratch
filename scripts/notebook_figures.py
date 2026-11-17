"""Editable, deterministic scientific illustrations for the PLUTO lesson.

Geometry, examples and dependency arrows describe the local implementation.
The introductory artwork is retained separately with its generation prompt.
"""

import base64
import math
from pathlib import Path
from xml.sax.saxutils import escape

NAVY = "#172d46"
MUTED = "#52667c"
TEAL = "#007f80"
GOLD = "#b87508"
RED = "#bd4652"
BLUE = "#406dbe"
PALE = "#edf4f6"
LINE = "#d6e1e8"
ROOT = Path(__file__).resolve().parents[1]

# Stable cell anchors, learning questions and implementation sources, inventoried
# before rendering. Each tuple supplies the accessible alt text and caption too.
INVENTORY = {
    "overview": (
        "intro",
        "How does a scene become a selected plan?",
        "pluto/model.py",
        "A driving scene is converted to vector tokens, decoded into candidates, then selected; the pink branch shows training supervision.",
    ),
    "data-flow": (
        "data",
        "Which adjacent history pairs remain valid?",
        "pluto/data.py",
        "A missing observation invalidates both adjacent pairs. Future labels remain separate from the historical inputs.",
    ),
    "history-pyramid": (
        "history",
        "How do short and long temporal contexts combine?",
        "pluto/model.py",
        "Three temporal scales exchange context through projection, linear upsampling and addition; a 5-by-5 mask shows a window of three.",
    ),
    "scene-tokens": (
        "scene",
        "How do different object types share a feature width?",
        "pluto/model.py",
        "An analytical max-pool, geometric Fourier address and typed scene-token sequence illustrate the three representation steps.",
    ),
    "query-grid": (
        "arch",
        "Which queries interact in each attention stage?",
        "pluto/model.py",
        "Reference attention connects rows, mode attention connects columns, and cross-attention reads valid scene tokens.",
    ),
    "shapes": (
        "assembly",
        "Where do the batch, route, mode and time axes go?",
        "pluto/model.py",
        "One highlighted query produces a 4-by-6 plan; classification selects among six candidates or falls back to the free head.",
    ),
    "loss": (
        "objective",
        "How does distance to unsafe space penalize a plan?",
        "pluto/losses.py",
        "Vehicle circle centers sample a distance field. The single-circle example gives max(0,1.0+0.2−0.7)=0.5 metres.",
    ),
    "optimizer": (
        "optimization",
        "What happens between a loss and a parameter update?",
        "pluto/losses.py",
        "A Smooth L1 example explains the gradient direction; the adjacent sequence follows the notebook's actual AdamW call order.",
    ),
    "contrastive": (
        "cil",
        "Which changed views can still use their old targets?",
        "pluto/objective.py",
        "All views share an encoder. Only original and positive views are decoded for supervision; the negative contributes to contrastive learning.",
    ),
    "inference": (
        "evaluation",
        "How are selection and geometric error different?",
        "pluto/planner.py",
        "Toy logits select candidate zero; a separate two-point example computes ADE=(0+1)/2=0.5 metres.",
    ),
    "bicycle": (
        "simulation",
        "How does a controller track a reference plan?",
        "pluto/simulation.py",
        "Wheelbase and steering define the bicycle update. A 0.1-second straight step at 1 m/s advances by exactly 0.1 metres.",
    ),
    "checkpoint": (
        "checkpoint",
        "What must match before training can resume?",
        "pluto/train.py",
        "Training state is written to a temporary file and atomically replaced; configuration, data and implementation hashes guard resume.",
    ),
}


class Canvas:
    def __init__(self, number, title, subtitle, height=830):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1440" height="{height}" viewBox="0 0 1440 {height}">',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#52667c"/></marker><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0 8L8 0" stroke="#b9c7d2" stroke-width="1"/></pattern></defs>',
            f'<rect width="1440" height="{height}" fill="#fafbf9"/>',
            '<rect width="1440" height="8" fill="#007f80"/>',
        ]
        self.text(38, 51, f"PLUTO / {number:02d}", 17, TEAL, True)
        self.text(38, 96, title, 32, NAVY, True)
        self.text(38, 130, subtitle, 19, MUTED)

    def text(self, x, y, value, size=21, color=NAVY, bold=False, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=12, **attrs):
        extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {extra}/>'
        )

    def path(self, d, color=TEAL, width=3, dash=None, arrow=False, fill="none"):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{extra}/>'
        )

    def circle(self, x, y, r, fill=TEAL, stroke="none", **attrs):
        extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" {extra}/>'
        )

    def panel(self, x, y, w, h, label):
        self.rect(x, y, w, h)
        self.text(x + 22, y + 34, label, 21, NAVY, True)

    def note(self, text):
        self.path(f"M38 {self.height - 61} H1402", LINE, 1)
        self.text(38, self.height - 28, text, 17, MUTED)

    def car(self, x, y, color=TEAL, angle=0, opacity=1, scale=1):
        self.parts.append(
            f'<g transform="translate({x},{y}) rotate({angle}) scale({scale})" opacity="{opacity}">'
        )
        self.rect(-18, -33, 36, 66, color, NAVY, 8, stroke_width=1.5)
        self.rect(-13, -17, 26, 13, "#dceef0", "none", 3)
        self.rect(-12, 15, 24, 9, "#c8d8e1", "none", 2)
        self.path("M-16 -24 H-8 M8 -24 H16", "#f9d583", 3)
        self.parts.append("</g>")

    def road(self, x, y, w, h):
        self.rect(x, y, w, h, "#e7eee9", "none", 8)
        self.rect(x + w * 0.2, y, w * 0.6, h, "#e1e8ed", "none", 0)
        for ratio in (0.23, 0.77):
            self.path(f"M{x + w * ratio} {y} V{y + h}", "white", 3)
        self.path(f"M{x + w * 0.5} {y + 7} V{y + h - 7}", "white", 3, "14 12")

    def token(self, x, y, w, h, color=TEAL, cells=8):
        self.rect(x, y, w, h, "#f1f7f7", color, 5)
        for i in range(cells):
            value = 0.18 + ((i * 7 + 3) % 11) / 14
            self.rect(
                x + 4 + i * (w - 8) / cells,
                y + 4,
                (w - 8) / cells - 3,
                h - 8,
                color,
                "none",
                2,
                opacity=f"{value:.3f}",
            )

    def finish(self):
        return "".join(self.parts) + "</svg>"


def overview():
    c = Canvas(
        1,
        "From a driving scene to a family of possible futures",
        "PLUTO educational implementation • conceptual artwork above, computational dependencies below",
        1120,
    )
    image = base64.b64encode(
        (ROOT / "notebooks/assets/illustrations/pluto-scene-to-plan.png").read_bytes()
    ).decode()
    c.parts.append(
        f'<image x="38" y="153" width="1364" height="454" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/png;base64,{image}"/>'
    )
    for x, label in [(60, "01  OBSERVE"), (530, "02  REPRESENT"), (1035, "03  PLAN")]:
        c.text(x, 640, label, 20, TEAL, True)
    for x, title, detail in [
        (38, "Scene encoder", "history + map + state → scene tokens"),
        (506, "Factorized decoder", "reference × mode queries read the scene"),
        (974, "Trajectory + score heads", "candidate plans [B,R,M,T,6] + logits"),
    ]:
        c.panel(x, 674, 428, 111, title)
        c.text(x + 20, 747, detail, 17)
    c.path("M466 730 H506", MUTED, 2, arrow=True)
    c.path("M934 730 H974", MUTED, 2, arrow=True)
    c.rect(38, 818, 428, 83, "#edf5f7")
    c.text(58, 850, "Reference geometry + learned modes", 19, NAVY, True)
    c.text(58, 879, "[B,R,P,6] + [M,D] → [B,R,M,D]", 19)
    c.path("M466 860 H486 V815 Q501 804 486 793 V790 H850 V785", MUTED, 2, arrow=True)
    c.rect(506, 818, 428, 83, "#edf2f9")
    c.text(526, 850, "Scene-token heads", 19, BLUE, True)
    c.text(526, 879, "Other-agent forecasts + free ego plan", 17)
    c.path("M250 785 V804 H720 V818", MUTED, 2, arrow=True)
    c.rect(974, 818, 428, 83, "#e5f4ef")
    c.text(994, 850, "Inference: selected [B,T,6]", 19, TEAL, True)
    c.text(994, 879, "No valid reference → separate free plan", 17)
    c.path("M1188 785 V818", MUTED, 2, arrow=True)
    c.path("M934 860 H974", MUTED, 2, arrow=True)
    c.rect(38, 940, 1364, 91, "#fbeff0")
    c.text(60, 975, "TRAINING ONLY", 18, RED, True)
    c.text(
        255,
        975,
        "Ground-truth futures + masks → imitation / prediction losses; optional SDF + contrastive views",
        18,
    )
    c.text(
        60,
        1009,
        "Loss gradients update the encoder, decoder and heads. Generated road art is schematic; exact tensors are worked below.",
        17,
    )
    c.path("M1402 729 H1420 V984 H1402", RED, 2, "6 5", True)
    c.path("M720 901 V940", RED, 2, "6 5", True)
    c.note(
        "Paper-sized dimensions: D=128, M=12, T=80. This notebook executes a small CPU configuration with random weights."
    )
    return c.finish()


def data_flow():
    c = Canvas(
        2,
        "A history becomes motion features — with missing data kept visible",
        "One worked mask example • current ego frame • metres and metres per second",
    )
    c.panel(38, 162, 645, 290, "A / Observations in the current ego frame")
    c.path("M100 385 H625", MUTED, 2, arrow=True)
    c.path("M100 385 V235", MUTED, 2, arrow=True)
    c.text(585, 419, "x (m)", 17)
    c.text(67, 229, "y (m)", 17)
    coords = [(145, 363), (244, 337), (343, 301), (442, 266), (541, 230)]
    c.path("M145 363 L244 337 L343 301 L442 266 L541 230", TEAL, 2, "6 6")
    for i, (x, y) in enumerate(coords):
        c.circle(x, y, 10, "white" if i == 3 else TEAL, TEAL, stroke_width=2)
        if i == 3:
            c.path(f"M{x - 6} {y - 6} l12 12 m-12 0 l12 -12", RED, 2)
        c.text(x, y + 35, f"t{i}", 18, anchor="middle")
    c.text(117, 222, "Past", 17, MUTED)
    c.text(510, 205, "Now", 17, TEAL, True)
    c.panel(713, 162, 689, 290, "B / Adjacent observations must both exist")
    for row, values in enumerate(((1, 1, 1, 0, 1), (1, 1, 0, 0))):
        y = 244 + row * 91
        c.text(735, y + 27, "valid" if row == 0 else "pair", 18, MUTED)
        for i, v in enumerate(values):
            x = 830 + i * 105
            c.rect(
                x, y, 78, 47, "#dff1eb" if v else "url(#hatch)", TEAL if v else LINE, 6
            )
            c.text(x + 39, y + 31, str(v), 22, TEAL if v else MUTED, True, "middle")
            if row == 0:
                c.text(x + 39, y - 12, f"t{i}", 17, anchor="middle")
    c.text(735, 425, "pair[i] = valid[i] AND valid[i+1]", 20, NAVY, True)
    c.panel(38, 479, 831, 272, "C / Each valid pair produces 9 motion features")
    for i, (label, n) in enumerate(
        (
            ("Δ position", 2),
            ("Δ velocity", 2),
            ("cos / sin Δθ", 2),
            ("dimensions", 2),
            ("valid", 1),
        )
    ):
        x = 62 + i * 158
        c.text(x, 560, label, 17, MUTED)
        for j in range(n):
            c.rect(x + j * 54, 580, 46, 54, "#e2eff5" if i < 2 else "#e5f3ed", LINE, 5)
        c.text(x, 667, f"{n} channel" + ("s" if n > 1 else ""), 17)
    c.text(
        62, 718, "[B,A,H,9]  →  valid mask [B,A,H]  →  history encoder", 21, TEAL, True
    )
    c.panel(897, 479, 505, 272, "D / Future labels are separate")
    c.text(922, 557, "target [B,T,6]", 24, TEAL, True)
    c.text(922, 601, "x, y, cos θ, sin θ, vx, vy", 21)
    c.text(922, 652, "Observed history → input", 20)
    c.text(922, 693, "Future trajectory → supervision", 20, RED)
    c.note(
        "Toy validity [1,1,1,0,1] gives pair validity [1,1,0,0]. Padded values must be finite; True always means valid."
    )
    return c.finish()


def history():
    c = Canvas(
        3,
        "Local attention gathers motion at three time scales",
        "Temporal pyramid + top-down fusion • shape example D=128 and 20 history vectors",
    )
    c.panel(
        38,
        164,
        908,
        584,
        "A / Downsample in time, then return coarse context to fine time",
    )
    for row, (count, label) in enumerate(
        ((20, "20 × 32"), (10, "10 × 64"), (5, "5 × 128"))
    ):
        y = 255 + row * 154
        c.text(61, y - 22, label, 21, NAVY, True)
        for i in range(count):
            c.rect(
                190 + i * 24,
                y - 45,
                18,
                58,
                "#007f80",
                "none",
                2,
                opacity=f"{0.3 + 0.7 * ((i % 5) / 4):.2f}",
            )
        c.text(192, y + 45, "2 local-attention blocks", 17)
        c.path(f"M{195 + count * 24} {y - 13} H755", MUTED, 2, arrow=True)
        c.token(766, y - 44, 145, 58, BLUE)
        c.text(622, y + 45, "project to D", 17, BLUE)
        if row < 2:
            c.path(f"M251 {y + 57} V{y + 88}", MUTED, 2, arrow=True)
            c.text(270, y + 82, "stride-2 convolution", 17, MUTED)
        if row > 0:
            c.path(f"M838 {y - 48} V{y - 119}", BLUE, 3, arrow=True)
            c.circle(838, y - 141, 13, "white", BLUE, stroke_width=2)
            c.text(838, y - 135, "+", 20, BLUE, True, "middle")
            c.text(852, y - 83, "↑", 21, BLUE)
    c.text(
        61,
        712,
        "Last valid fine-scale feature → one agent token [B,A,128]",
        21,
        TEAL,
        True,
    )
    c.panel(975, 164, 427, 584, "B / What can timestep 2 read?")
    c.text(1000, 238, "Window = 3; toy length = 5", 18)
    for i in range(5):
        c.text(1084 + i * 57, 279, str(i), 17, anchor="middle")
        c.text(1022, 324 + i * 57, str(i), 17)
        for j in range(5):
            allowed = abs(i - j) <= 1
            c.rect(
                1060 + j * 57,
                294 + i * 57,
                47,
                47,
                "#007f80" if allowed else "url(#hatch)",
                "none",
                4,
                opacity=1 if i == 2 else 0.45,
            )
    c.text(1000, 627, "row 2 → keys {1,2,3}", 20, TEAL, True)
    c.text(1000, 666, "Invalid keys are also excluded.", 18)
    c.text(1000, 704, "Residual + MLP at each block.", 18)
    c.note(
        "Blue arrows: linear upsampling + addition. Empty agents return zero. This local operator has no established NATTEN parity."
    )
    return c.finish()


def scene_tokens():
    c = Canvas(
        4,
        "Different scene objects become a common token language",
        "Point pooling, ego-state attention and Fourier pose features in the local scene encoder",
    )
    c.panel(38, 164, 490, 579, "A / A polyline becomes one token")
    pts = [(90, 376), (160, 323), (240, 301), (328, 252), (436, 241)]
    c.path("M90 376 L160 323 L240 301 L328 252 L436 241", TEAL, 3)
    for i, (x, y) in enumerate(pts):
        c.circle(x, y, 8, TEAL)
        c.text(x, y - 18, f"p{i}", 16, TEAL, anchor="middle")
    c.text(63, 429, "Pointwise MLP → masked max over points", 18)
    vals = ((0.2, 0.8, 0.4), (0.9, 0.1, 0.3), (0.5, 0.4, 0.7))
    for row, values in enumerate(vals):
        for col, value in enumerate(values):
            c.rect(
                90 + col * 80, 452 + row * 48, 65, 38, TEAL, "none", 4, opacity=value
            )
            c.text(
                122 + col * 80,
                478 + row * 48,
                value,
                17,
                "white" if value > 0.6 else NAVY,
                anchor="middle",
            )
    c.path("M340 519 H385", MUTED, 2, arrow=True)
    for i, value in enumerate((0.9, 0.8, 0.7)):
        c.text(426, 478 + i * 48, value, 20, TEAL, True, "middle")
    c.text(63, 635, "Toy pooled feature: [0.9, 0.8, 0.7]", 20, TEAL, True)
    c.text(63, 688, "All points missing → zero token.", 18)
    c.panel(552, 164, 435, 579, "B / Attach a geometric address")
    c.path("M595 371 H927 M595 371 V243", MUTED, 2, arrow=True)
    c.car(746, 298, TEAL, 35, scale=0.8)
    c.path("M746 298 l54 -39", TEAL, 2, arrow=True)
    c.text(809, 257, "θ", 20, TEAL, True)
    c.text(624, 409, "pose = (x, y, θ)", 23, TEAL, True)
    for i in range(3):
        d = " ".join(
            f"{'M' if j == 0 else 'L'}{598 + j * 6:.1f} {478 + i * 58 + 16 * math.sin(j * (i + 1) * 0.25):.2f}"
            for j in range(57)
        )
        c.path(d, (TEAL, GOLD, BLUE)[i], 2)
    c.text(578, 680, "sin/cos frequencies → pose embedding", 17)
    c.panel(1011, 164, 391, 579, "C / Assemble the scene")
    for i, (label, color) in enumerate(
        (
            ("ego history + state", TEAL),
            ("dynamic agents", BLUE),
            ("map polylines", GOLD),
            ("static objects", RED),
        )
    ):
        y = 239 + i * 95
        c.text(1033, y, label, 19, color, True)
        c.token(1033, y + 12, 237, 36, color)
    c.text(1033, 647, "+ pose and type embeddings", 18)
    c.text(1033, 690, "scene transformer → [B,N,D]", 18, TEAL, True)
    c.note(
        "Heatmap and wave amplitudes are illustrative. The toy max-pooling values are analytical; all object types share width D."
    )
    return c.finish()


def queries():
    c = Canvas(
        5,
        "Factorized attention turns reference lines × modes into plans",
        "Toy grid R=2, M=3 • each tile is a D-dimensional query, not a trajectory coordinate",
    )
    for px, label in (
        (38, "A / Across references"),
        (507, "B / Across modes"),
        (976, "C / Read the scene"),
    ):
        c.panel(px, 164, 426, 424, label)
        for r in range(2):
            for m in range(3):
                x, y = px + 67 + m * 100, 295 + r * 114
                c.rect(x, y, 77, 61, "#e4f3ee", TEAL, 8)
                c.text(x + 38, y + 38, f"q{r}{m}", 21, TEAL, True, "middle")
        for m in range(3):
            c.text(px + 105 + m * 100, 270, f"m{m}", 17, anchor="middle")
        for r in range(2):
            c.text(px + 17, 333 + r * 114, f"r{r}", 17)
        if px == 38:
            for m in range(3):
                c.path(f"M{px + 106 + m * 100} 362 V401", BLUE, 3, arrow=True)
        elif px == 507:
            for r in range(2):
                c.path(f"M{px + 147} {325 + r * 114} H{px + 158}", GOLD, 3, arrow=True)
                c.path(f"M{px + 247} {325 + r * 114} H{px + 258}", GOLD, 3, arrow=True)
        else:
            for i in range(7):
                c.circle(px + 44 + i * 53, 226, 9, (TEAL, BLUE, GOLD)[i % 3])
            for m in range(3):
                c.path(f"M{px + 105 + m * 100} 289 V250", MUTED, 1.5, arrow=True)
        c.path(f"M{px + 55} 294 V488 H{px + 368}", BLUE, 1.7, arrow=True)
        c.circle(px + 382, 488, 12, "white", BLUE, stroke_width=2)
        c.text(px + 382, 495, "+", 20, BLUE, True, "middle")
        c.text(px + 135, 510, "residual addition", 15, BLUE)
        c.text(
            px + 23,
            556,
            (
                "Same mode, different lateral route."
                if px == 38
                else "Same route, different learned mode."
                if px == 507
                else "Scene keys use their valid masks."
            ),
            17,
        )
    c.path("M464 384 H507", MUTED, 2, arrow=True)
    c.path("M933 384 H976", MUTED, 2, arrow=True)
    c.rect(38, 615, 1364, 128, "#eaf2f5")
    c.text(61, 654, "Build", 19, NAVY, True)
    c.text(61, 691, "q[r,m] = Linear(concat(ref[r], mode[m]))", 21, TEAL, True)
    c.text(746, 654, "Attention work, ignoring width D", 19, NAVY, True)
    c.text(746, 691, "R²M + RM² = 12 + 18 = 30; full grid: 36", 21)
    c.note(
        "Every attention sublayer has a residual route; feedforward follows. Modes are learned embeddings, not fixed speed bins at inference."
    )
    return c.finish()


def shapes():
    c = Canvas(
        6,
        "Trace one query all the way to a multichannel trajectory",
        "Toy batch B=2, R=2, M=3, T=4, D=16 • each output timestep has 6 channels",
    )
    c.panel(38, 164, 482, 579, "A / Six candidate plans per scene")
    for r in range(2):
        for m in range(3):
            x, y = 83 + m * 137, 256 + r * 170
            c.rect(
                x,
                y,
                111,
                124,
                "#f1f7f7",
                TEAL if (r, m) == (1, 2) else LINE,
                7,
                stroke_width=3 if (r, m) == (1, 2) else 1,
            )
            c.text(x + 13, y + 26, f"r{r}, m{m}", 18, TEAL, True)
            c.path(
                f"M{x + 16} {y + 105} Q{x + 35} {y + 60} {x + 89} {y + 47 + m * 8}",
                TEAL if (r, m) == (1, 2) else MUTED,
                2,
            )
    c.text(62, 657, "trajectory [2,2,3,4,6]", 23, TEAL, True)
    c.text(62, 699, "The example highlights reference 1, mode 2.", 18)
    c.panel(544, 164, 858, 289, "B / One candidate: 4 timesteps × 6 channels")
    labels = ["x", "y", "cos θ", "sin θ", "vx", "vy"]
    for j, label in enumerate(labels):
        c.text(675 + j * 119, 252, label, 18, anchor="middle")
    for t in range(4):
        c.text(570, 292 + t * 42, f"t{t + 1}", 18, MUTED)
        for j in range(6):
            c.rect(
                632 + j * 119,
                269 + t * 42,
                85,
                31,
                TEAL if j < 2 else "#dce7ed",
                "none",
                3,
                opacity=0.6 + 0.1 * t if j < 2 else 1,
            )
    c.panel(544, 479, 858, 264, "C / Candidate classification and safe fallback")
    c.text(569, 563, "logits [2,2,3] → argmax over 6 candidates", 23, TEAL, True)
    c.path("M570 585 H1363", LINE, 1)
    c.text(569, 626, "Valid reference exists", 20, NAVY, True)
    c.text(1038, 626, "selected [2,4,6]", 20, TEAL, True)
    c.path("M889 619 H1008", MUTED, 2, arrow=True)
    c.text(569, 691, "No valid reference", 20, NAVY, True)
    c.text(1038, 691, "free [2,4,6]", 20, RED, True)
    c.path("M889 684 H1008", MUTED, 2, arrow=True)
    c.note(
        "Each plan has 4 × 2 = 8 position values, plus heading and velocity. Other-agent forecasts use a separate prediction head."
    )
    return c.finish()


def loss():
    c = Canvas(
        7,
        "Three-circle geometry turns a distance field into a planning penalty",
        "Worked single-circle hinge example • distance is positive in safe space • optional auxiliary term",
    )
    c.panel(38, 164, 682, 578, "A / Sample the field at vehicle circle centers")
    for i in range(10):
        for j in range(12):
            value = min(1.0, max(0, (j - 1) / 8))
            color = RED if j < 3 else TEAL
            c.rect(
                64 + j * 52,
                228 + i * 40,
                50,
                38,
                color,
                "none",
                2,
                opacity=0.1 + abs(value - 0.25) * 0.45,
            )
    c.path("M194 231 V622", RED, 4)
    c.text(73, 671, "obstacle", 18, RED, True)
    c.text(459, 671, "safe-space distance →", 18, TEAL, True)
    c.car(395, 428, TEAL, 0, scale=2.1)
    for y in (359, 428, 497):
        c.circle(395, y, 67, "none", GOLD, stroke_width=2.5, stroke_dasharray="7 5")
        c.circle(395, y, 4, GOLD)
    c.path("M395 428 H199", GOLD, 3, arrow=True)
    c.text(254, 407, "d", 25, GOLD, True)
    c.text(460, 474, "r", 25, GOLD, True)
    c.panel(745, 164, 657, 299, "B / One numerical penalty")
    c.text(771, 250, "radius r = 1.0 m    margin = 0.2 m", 22)
    c.text(771, 297, "sampled distance d = 0.7 m", 22)
    c.rect(770, 322, 605, 91, "#fff2da", "none")
    c.text(792, 358, "max(0, r + margin − d)", 23, GOLD, True)
    c.text(792, 392, "= max(0, 1.0 + 0.2 − 0.7) = 0.5 m", 22)
    c.panel(745, 489, 657, 253, "C / How the gradient reaches the model")
    c.text(771, 566, "trajectory → circle centers → bilinear lookup", 20)
    c.path("M782 595 H1358", RED, 2, "6 4", True)
    c.text(771, 638, "hinge → ∂loss/∂position → model parameters", 20, RED)
    c.text(771, 694, "Sum circle costs; average valid timesteps.", 20, TEAL, True)
    c.note(
        "The displayed field is illustrative, not a measured raster. The 0.5 m one-circle result is checked by the adjacent executable probe."
    )
    return c.finish()


def optimizer():
    c = Canvas(
        8,
        "A training step: compute, differentiate, then update",
        "A concrete Smooth L1 example plus the actual order of the notebook's AdamW step",
    )
    c.panel(38, 164, 650, 399, "A / Inspect one scalar regression term")
    c.path("M103 493 H628 M103 493 V252", MUTED, 2, arrow=True)
    c.text(528, 535, "prediction x", 17)
    c.text(117, 245, "Smooth L1 loss; target x = 1", 18, TEAL, True)
    points = []
    for i in range(81):
        x = i / 20 - 1
        d = abs(x - 1)
        value = 0.5 * d * d if d < 1 else d - 0.5
        points.append(
            f"{'M' if i == 0 else 'L'}{110 + i * 6:.2f} {490 - value * 90:.2f}"
        )
    c.path(" ".join(points), TEAL, 3)
    for value in (0, 1, 2):
        c.text(230 + value * 120, 518, str(value), 16, MUTED, anchor="middle")
    c.circle(230, 445, 7, GOLD)
    c.text(250, 427, "x=0 → loss=0.5", 21, GOLD, True)
    c.path("M230 458 H300", RED, 3, arrow=True)
    c.text(180, 321, "gradient = −1", 20, RED, True)
    c.text(180, 357, "descent moves x toward 1", 19)
    c.panel(715, 164, 687, 579, "B / Execute in dependency order")
    steps = [
        ("1", "forward(batch)", "plans, logits, forecasts, free trajectory"),
        ("2", "compute losses", "sum channels; mask invalid timesteps"),
        ("3", "zero_grad()", "clear the previous accumulated gradients"),
        ("4", "loss.backward()", "autograd follows the dependency graph"),
        ("5", "optimizer.step()", "AdamW updates learnable parameters"),
    ]
    for i, (number, label, detail) in enumerate(steps):
        y = 238 + i * 93
        c.circle(752, y + 8, 20, TEAL if i != 3 else RED)
        c.text(752, y + 15, number, 19, "white", True, "middle")
        c.text(791, y + 3, label, 22, NAVY, True)
        c.text(791, y + 33, detail, 17)
        if i < 4:
            c.path(f"M752 {y + 32} V{y + 62}", MUTED, 2, arrow=True)
    c.rect(38, 590, 650, 153, "#eaf3f4")
    c.text(62, 633, "Full notebook loss", 21, NAVY, True)
    c.text(62, 677, "chosen-plan regression + classification", 21, TEAL)
    c.text(62, 715, "+ free-plan regression + other-agent prediction", 20, TEAL)
    c.note(
        "The Smooth L1 example is analytical. AdamW uses moment estimates and weight decay, so the scalar arrow is not a measured step."
    )
    return c.finish()


def contrastive():
    c = Canvas(
        9,
        "Change a scene, then decide which supervision still makes sense",
        "Illustrative view construction • causal validity must be supplied by preprocessing",
    )
    for i, (title, detail) in enumerate(
        (
            ("ORIGINAL", "all observed actors"),
            ("POSITIVE", "irrelevant actor hidden"),
            ("NEGATIVE", "relevant actor changed"),
        )
    ):
        x = 38 + i * 465
        c.panel(x, 164, 434, 388, title)
        c.road(x + 24, 220, 386, 255)
        c.car(x + 260, 425, TEAL, scale=0.85)
        c.car(x + 260, 292 if i != 2 else 354, RED if i == 2 else NAVY, scale=0.75)
        if i != 1:
            c.car(x + 130, 265, BLUE, 180, scale=0.7)
        else:
            c.circle(x + 130, 265, 28, "none", BLUE, stroke_dasharray="5 4")
        c.path(f"M{x + 260} 397 V371", TEAL, 3, arrow=True)
        c.text(x + 22, 519, detail, 19, TEAL if i < 2 else RED, True)
    for i in range(3):
        c.path(f"M{255 + i * 465} 552 V579 H720 V602", MUTED, 2, arrow=i == 1)
    c.rect(38, 606, 610, 136, "#e8f4ee")
    c.text(63, 643, "Supervised branch", 22, TEAL, True)
    c.text(63, 681, "Decode original + positive; use their labels.", 20)
    c.text(63, 719, "Negative: encode only; old target is unused.", 19, RED)
    c.rect(674, 606, 728, 136, "#edf2f9")
    c.text(
        699, 643, "Shared encoder → projection → normalized embeddings", 20, BLUE, True
    )
    c.text(699, 685, "similarity(anchor, positive) ↑", 23, TEAL, True)
    c.text(699, 721, "similarity(anchor, negative) ↓  via 2-way cross-entropy", 19, RED)
    c.note(
        "Scene drawings explain the intended idea. The notebook uses synthetic masks and does not certify causal correctness of any real scene."
    )
    return c.finish()


def inference():
    c = Canvas(
        10,
        "Separate candidate selection from trajectory error",
        "The notebook uses learned logits; the package also exposes a caller-supplied rule-scoring interface",
    )
    c.panel(38, 164, 651, 578, "A / Toy learned selection with two candidates")
    c.road(81, 230, 282, 449)
    c.car(260, 623, TEAL)
    c.path("M260 583 C260 498 250 382 252 269", TEAL, 5, arrow=True)
    c.path("M260 583 C250 465 159 393 150 310", GOLD, 3, "8 5", True)
    c.text(390, 277, "logits", 20, MUTED, True)
    for i, (name, value, color) in enumerate(
        (("candidate 0", 2.0, TEAL), ("candidate 1", 1.0, GOLD))
    ):
        y = 335 + i * 112
        c.text(390, y, name, 20, color, True)
        c.rect(390, y + 18, value * 110, 26, color, "none", 3)
        c.text(390, y + 75, str(value), 20, color)
    c.text(390, 595, "argmax → 0", 24, TEAL, True)
    c.text(390, 642, "Invalid references", 18)
    c.text(390, 674, "are excluded.", 18)
    c.panel(716, 164, 686, 381, "B / Open-loop ADE has a geometric meaning")
    c.path("M780 447 H1322 M780 447 V260", MUTED, 2, arrow=True)
    c.text(1290, 486, "x (m)", 17)
    c.path("M820 409 H1050", TEAL, 4)
    c.path("M820 409 H1280", GOLD, 3, "7 5")
    for x in (820, 1050):
        c.circle(x, 409, 8, TEAL)
    c.circle(1280, 409, 8, GOLD)
    c.text(815, 365, "0", 18)
    c.text(1045, 365, "1", 18)
    c.text(1275, 365, "2", 18)
    c.text(783, 250, "Target: (0,0), (1,0)", 20, TEAL, True)
    c.text(783, 288, "Prediction: (0,0), (2,0)", 20, GOLD, True)
    c.text(783, 518, "ADE = (0 m + 1 m) / 2 = 0.5 m", 23, TEAL, True)
    c.rect(716, 572, 686, 170, "#f1f4f8")
    c.text(740, 614, "Optional package-level selection", 21, NAVY, True)
    c.text(740, 655, "top-K → external evaluator → rule + α·probability", 20)
    c.text(740, 704, "Returns the original proposal; no references → free plan.", 18)
    c.note(
        "ADE is an open-loop geometric diagnostic. The official nuPlan closed-loop score requires separate simulation and evaluation."
    )
    return c.finish()


def bicycle():
    c = Canvas(
        11,
        "A reference path becomes a physically tracked rollout",
        "Educational kinematic bicycle + finite-horizon LQR controller • state (x,y,θ,v), control (a,δ)",
    )
    c.panel(
        38, 164, 715, 578, "A / Steering creates heading change around the rear axle"
    )
    c.path("M105 633 H682 M105 633 V251", MUTED, 2, arrow=True)
    c.text(662, 670, "x (m)", 17)
    c.text(66, 244, "y (m)", 17)
    c.car(355, 439, TEAL, 60, scale=2.2)
    c.path("M287 479 L407 409", GOLD, 3)
    c.text(354, 510, "wheelbase L", 20, GOLD, True)
    c.path("M408 408 l64 -71", RED, 4, arrow=True)
    c.text(458, 386, "steering δ", 19, RED, True)
    c.path("M287 479 l100 -58", TEAL, 3, arrow=True)
    c.text(230, 405, "heading θ", 20, TEAL, True)
    c.path("M289 479 C401 409 466 302 550 241", TEAL, 3, "6 5")
    c.text(105, 711, "θ′ = θ + v·tan(δ)·Δt / L", 24, TEAL, True)
    c.panel(778, 164, 624, 282, "B / Verify one straight step")
    c.text(802, 249, "x=0, y=0, θ=0, v=1 m/s", 22)
    c.text(802, 295, "Δt=0.1 s, steering δ=0", 22)
    c.text(802, 354, "x′ = 0 + 1·cos(0)·0.1 = 0.1 m", 22, TEAL, True)
    c.text(802, 401, "y′ = 0 m; heading remains 0 rad", 20)
    c.panel(778, 474, 624, 268, "C / Controller computation")
    for i, (label, color) in enumerate(
        (
            ("Reference states", TEAL),
            ("Backward Riccati recursion → gains", BLUE),
            ("Forward bicycle rollout → tracked states", GOLD),
        )
    ):
        c.text(805, 552 + i * 71, label, 20, color, True)
        if i < 2:
            c.path(f"M827 {566 + i * 71} V{593 + i * 71}", MUTED, 2, arrow=True)
    c.note(
        "Only the straight-step arithmetic is numeric here. The local bicycle/LQR controller has not been validated against official nuPlan/PDM."
    )
    return c.finish()


def checkpoint():
    c = Canvas(
        12,
        "Recover an epoch boundary without silently changing the experiment",
        "Infrastructure workflow • atomic file replacement • exact resume scoped to tested CPU synthetic runs",
    )
    c.panel(38, 164, 652, 366, "A / What must survive a restart?")
    for i, (label, color) in enumerate(
        (
            ("model parameters", TEAL),
            ("optimizer moments", BLUE),
            ("scheduler + scaler", GOLD),
            ("RNG states + epoch", RED),
            ("config + data identity", TEAL),
            ("implementation SHA-256", BLUE),
        )
    ):
        col, row = i % 2, i // 2
        x, y = 64 + col * 308, 239 + row * 89
        c.token(x, y, 56, 42, color, 3)
        c.text(x + 70, y + 27, label, 17, NAVY, True)
    c.panel(718, 164, 684, 366, "B / Publish a complete checkpoint")
    for i, label in enumerate(
        ("Serialize state", "checkpoint.pt.tmp", "Atomic replace → checkpoint.pt")
    ):
        y = 249 + i * 87
        c.circle(755, y - 5, 15, TEAL)
        c.text(791, y + 3, label, 23, NAVY, True)
        if i < 2:
            c.path(f"M755 {y + 15} V{y + 59}", MUTED, 2, arrow=True)
    c.panel(38, 557, 1364, 185, "C / Resume only the same experiment")
    c.text(63, 648, "Check config / manifest / code hashes", 22, TEAL, True)
    c.path("M540 642 H604", MUTED, 2, arrow=True)
    c.text(625, 648, "Restore state", 22, BLUE, True)
    c.path("M824 642 H888", MUTED, 2, arrow=True)
    c.text(910, 648, "Continue at next epoch", 22, GOLD, True)
    c.text(
        63,
        706,
        "Mismatch or missing code hashes → reject resume before replacing run provenance.",
        21,
        RED,
    )
    c.note(
        "CUDA and real-data resume need their own validation. Optional MLflow and progress/ETA are implemented outside this notebook."
    )
    return c.finish()


DRAWERS = {
    "overview": overview,
    "data-flow": data_flow,
    "history-pyramid": history,
    "scene-tokens": scene_tokens,
    "query-grid": queries,
    "shapes": shapes,
    "loss": loss,
    "optimizer": optimizer,
    "contrastive": contrastive,
    "inference": inference,
    "bicycle": bicycle,
    "checkpoint": checkpoint,
}


def draw(fid):
    return DRAWERS[fid]()
