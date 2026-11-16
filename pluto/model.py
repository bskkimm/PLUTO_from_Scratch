"""Vector scene encoder and factorized lateral/longitudinal planner.

All spatial inputs are in the ego frame, in metres. Masks are True for valid
entries. The first agent is ego; other agent future predictions are agent-relative.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PlutoConfig:
    dim: int = 128
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    modes: int = 12
    future_steps: int = 80
    dropout: float = 0.1
    state_dropout: float = 0.75


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class PointEncoder(nn.Module):
    """Masked pointwise MLP and max pooling; empty polylines map to zero."""

    def __init__(self, channels, dim):
        super().__init__()
        self.net = mlp(channels, dim, dim)

    def forward(self, x, mask):
        y = self.net(x).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = y.max(dim=-2).values
        return torch.where(
            mask.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled)
        )


class LocalAttentionBlock(nn.Module):
    """Windowed temporal attention with a residual MLP and valid-key masks."""

    def __init__(self, dim, heads, window=3):
        super().__init__()
        if window < 1 or window % 2 != 1:
            raise ValueError("window must be a positive odd integer")
        self.window = window
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim)
        )

    def forward(self, x, valid):
        length = x.shape[1]
        offsets = torch.arange(length, device=x.device)
        outside = (offsets[:, None] - offsets[None, :]).abs() > self.window // 2
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(-1), 0] = True
        # A valid key may lie outside a padded query's window; grant padded
        # queries one dummy key, then zero their outputs immediately afterwards.
        expanded = outside.unsqueeze(0).expand(x.shape[0], -1, -1).clone()
        expanded |= ~safe_valid[:, None, :]
        empty = expanded.all(-1)
        expanded[empty] = False
        mask = expanded.repeat_interleave(self.attention.num_heads, dim=0)
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        value = self.norm1(x)
        x = (
            x
            + self.attention(value, value, value, attn_mask=mask, need_weights=False)[0]
        )
        x = x + self.feedforward(self.norm2(x))
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class HistoryEncoder(nn.Module):
    """Three-level local-attention pyramid for masked agent history vectors.

    Window boundaries and masking are implemented in PyTorch; this is not yet
    operator-compatible with the released NATTEN implementation.
    """

    def __init__(self, channels, dim, heads):
        super().__init__()
        if dim % 4:
            raise ValueError("history embedding dim must be divisible by four")
        dims = (dim // 4, dim // 2, dim)
        self.tokenizer = nn.Conv1d(channels, dims[0], 3, padding=1)
        self.levels = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LocalAttentionBlock(
                            width, math.gcd(width, heads), window=3 if i < 2 else 5
                        )
                        for _ in range(2)
                    ]
                )
                for i, width in enumerate(dims)
            ]
        )
        self.down = nn.ModuleList(
            [nn.Conv1d(dims[i], dims[i + 1], 3, stride=2, padding=1) for i in range(2)]
        )
        self.lateral = nn.ModuleList(
            [nn.Conv1d(width, dim, 3, padding=1) for width in dims]
        )
        self.fuse = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, history, valid):
        b, a, t, c = history.shape
        mask = valid.reshape(b * a, t)
        value = history.reshape(b * a, t, c).masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.tokenizer(value.transpose(1, 2)).transpose(1, 2)
        outputs = []
        for i, blocks in enumerate(self.levels):
            for block in blocks:
                x = block(x, mask)
            outputs.append(x.transpose(1, 2))
            if i < len(self.down):
                x = self.down[i](x.transpose(1, 2)).transpose(1, 2)
                mask = (
                    F.max_pool1d(mask.float().unsqueeze(1), 3, stride=2, padding=1)
                    .squeeze(1)
                    .bool()
                )
        levels = [layer(value) for layer, value in zip(self.lateral, outputs)]
        for i in range(len(levels) - 1, 0, -1):
            levels[i - 1] = levels[i - 1] + F.interpolate(
                levels[i],
                size=levels[i - 1].shape[-1],
                mode="linear",
                align_corners=False,
            )
        features = self.fuse(levels[0]).transpose(1, 2)
        original = valid.reshape(b * a, t)
        last = (
            torch.arange(t, device=history.device)
            .expand(b * a, t)
            .masked_fill(~original, -1)
            .max(-1)
            .values
        )
        result = features[torch.arange(b * a, device=history.device), last.clamp(min=0)]
        result = result.masked_fill((last < 0).unsqueeze(-1), 0.0)
        return result.reshape(b, a, -1)


class StateEncoder(nn.Module):
    """Attend to ego-state scalars; randomly hide dynamic scalars during training."""

    def __init__(self, dim, heads, dropout=0.75):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("state dropout must be in [0,1]")
        self.dropout = dropout
        self.embed = nn.ModuleList([nn.Linear(1, dim) for _ in range(6)])
        self.position = nn.Parameter(torch.randn(1, 6, dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, state):
        values = (
            torch.stack(
                [layer(state[:, i : i + 1]) for i, layer in enumerate(self.embed)],
                dim=1,
            )
            + self.position
        )
        mask = torch.zeros(state.shape[0], 6, dtype=torch.bool, device=state.device)
        if self.training and self.dropout:
            mask[:, 3:] = (
                torch.rand(state.shape[0], 3, device=state.device) < self.dropout
            )
        return self.attention(
            self.query.expand(state.shape[0], -1, -1),
            values,
            values,
            key_padding_mask=mask,
            need_weights=False,
        )[0][:, 0]


class FourierPosition(nn.Module):
    """Learned Fourier features for ego-frame (x, y, heading) poses."""

    def __init__(self, dim, bands=16):
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(3, bands))
        self.project = mlp(3 * (2 * bands + 1), dim, dim)

    def forward(self, pose):
        phase = pose.unsqueeze(-1) * self.frequency * (2 * math.pi)
        features = torch.cat((pose.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1)
        return self.project(features.flatten(-2))


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.lateral = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.longitudinal = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memory, ref_valid, scene_valid):
        b, r, m, d = q.shape
        x = q.transpose(1, 2).reshape(b * m, r, d)
        # An all-masked row yields NaNs in attention; a temporary dummy is ignored later.
        valid = ref_valid.clone()
        valid[~valid.any(-1), 0] = True
        mask = (~valid)[:, None, :].expand(b, m, r).reshape(b * m, r)
        y = self.norms[0](x)
        x = x + self.drop(
            self.lateral(y, y, y, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x.reshape(b, m, r, d).transpose(1, 2).reshape(b * r, m, d)
        y = self.norms[1](x)
        x = x + self.drop(self.longitudinal(y, y, y, need_weights=False)[0])
        x = x.reshape(b, r * m, d)
        y = self.norms[2](x)
        x = x + self.drop(
            self.cross(
                y, memory, memory, key_padding_mask=~scene_valid, need_weights=False
            )[0]
        )
        x = x + self.drop(self.ff(self.norms[3](x)))
        return x.reshape(b, r, m, d)


class Pluto(nn.Module):
    """PLUTO-style model. See docs/paper-to-code.md for deliberate differences."""

    def __init__(self, config=PlutoConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.agent_encoder = HistoryEncoder(9, d, config.heads)
        self.map_encoder = PointEncoder(10, d)
        self.static_encoder = mlp(5, d, d)
        self.ego_encoder = StateEncoder(d, config.heads, config.state_dropout)
        self.type_embedding = nn.Embedding(3, d)
        self.agent_type = nn.Embedding(4, d)
        self.map_type = nn.Embedding(3, d)
        self.route_type = nn.Embedding(2, d)
        self.traffic_type = nn.Embedding(4, d)
        self.static_type = nn.Embedding(4, d)
        self.speed_embedding = mlp(1, d, d)
        self.unknown_speed = nn.Parameter(torch.zeros(d))
        self.scene_position = FourierPosition(d)
        self.reference_position = FourierPosition(d)
        layer = nn.TransformerEncoderLayer(
            d, config.heads, d * 4, config.dropout, batch_first=True, norm_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(
            layer, config.encoder_layers, norm=nn.LayerNorm(d)
        )
        self.reference_encoder = PointEncoder(6, d)
        self.mode = nn.Parameter(torch.randn(config.modes, d) * 0.01)
        self.query_projection = nn.Linear(2 * d, d)
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d, config.heads, config.dropout)
                for _ in range(config.decoder_layers)
            ]
        )
        self.trajectory_head = mlp(d, 2 * d, config.future_steps * 6)
        self.score_head = mlp(d, d, 1)
        self.prediction_head = mlp(d, 2 * d, config.future_steps * 6)
        self.free_head = mlp(d, 2 * d, config.future_steps * 6)
        self.projection = mlp(d, d, d)

    def encode_scene(self, batch):
        """Encode scene tokens; contrastive negatives need no trajectory decoder."""
        agents, maps, static = batch["agents"], batch["map"], batch["static"]
        b, a = agents.shape[:2]
        agent_valid = batch["agent_mask"].any(-1)
        # The ego state supplies a valid scene token even if its history is absent.
        agent_valid = agent_valid.clone()
        agent_valid[:, 0] = True
        agent = self.agent_encoder(agents, batch["agent_mask"])
        agent = torch.cat(
            [self.ego_encoder(batch["ego_state"]).unsqueeze(1), agent[:, 1:]], dim=1
        )
        map_count, static_count = maps.shape[1], static.shape[1]
        default_agents = torch.zeros(b, a, dtype=torch.long, device=agents.device)
        default_map = torch.zeros(b, map_count, dtype=torch.long, device=maps.device)
        default_static = torch.zeros(
            b, static_count, dtype=torch.long, device=static.device
        )
        agent = agent + self.agent_type(batch.get("agent_type", default_agents).long())
        map_embedding = (
            self.map_encoder(maps, batch["map_mask"])
            + self.map_type(batch.get("map_type", default_map).long())
            + self.route_type(batch.get("on_route", default_map).long())
            + self.traffic_type(batch.get("traffic_light", default_map).long())
        )
        speed = batch.get("speed_limit", maps.new_zeros(b, map_count))
        known = batch.get(
            "speed_known",
            torch.zeros(b, map_count, dtype=torch.bool, device=maps.device),
        )
        map_embedding = map_embedding + torch.where(
            known.unsqueeze(-1),
            self.speed_embedding(speed.unsqueeze(-1)),
            self.unknown_speed,
        )
        static_embedding = self.static_encoder(static) + self.static_type(
            batch.get("static_type", default_static).long()
        )
        tokens = torch.cat(
            [
                agent + self.type_embedding.weight[0],
                map_embedding + self.type_embedding.weight[1],
                static_embedding + self.type_embedding.weight[2],
            ],
            dim=1,
        )
        scene_valid = torch.cat(
            [agent_valid, batch["map_mask"].any(-1), batch["static_mask"]], dim=1
        )
        agent_pose = batch.get("agent_pose", agents.new_zeros(b, a, 3))
        map_pose = batch.get("map_pose", maps.new_zeros(b, maps.shape[1], 3))
        static_pose = batch.get("static_pose", static[..., :3])
        pose = torch.cat((agent_pose, map_pose, static_pose), dim=1)
        tokens = tokens + self.scene_position(pose)
        memory = self.scene_encoder(tokens, src_key_padding_mask=~scene_valid)
        return memory, scene_valid

    def forward(self, batch):
        c = self.config
        b, a = batch["agents"].shape[:2]
        memory, scene_valid = self.encode_scene(batch)
        prediction = self.prediction_head(memory[:, 1:a]).reshape(
            b, a - 1, c.future_steps, 6
        )
        free = self.free_head(memory[:, 0]).reshape(b, c.future_steps, 6)
        refs, ref_mask = batch["reference"], batch["reference_mask"]
        r = refs.shape[1]
        if r == 0:
            return {
                "trajectory": None,
                "logits": None,
                "prediction": prediction,
                "free": free,
                "selected": free,
                "hidden": self.projection(memory[:, 0]),
            }
        ref_valid = ref_mask.any(-1)
        ref_pose = batch.get("reference_pose", refs.new_zeros(b, r, 3))
        ref = self.reference_encoder(refs, ref_mask) + self.reference_position(ref_pose)
        q = self.query_projection(
            torch.cat(
                [
                    ref.unsqueeze(2).expand(b, r, c.modes, c.dim),
                    self.mode[None, None].expand(b, r, c.modes, c.dim),
                ],
                -1,
            )
        )
        for layer in self.decoder:
            q = layer(q, memory, ref_valid, scene_valid)
        trajectory = self.trajectory_head(q).reshape(b, r, c.modes, c.future_steps, 6)
        logits = (
            self.score_head(q).squeeze(-1).masked_fill(~ref_valid.unsqueeze(-1), -1e9)
        )
        best = logits.reshape(b, -1).argmax(-1)
        selected = trajectory.reshape(b, r * c.modes, c.future_steps, 6)[
            torch.arange(b, device=best.device), best
        ]
        selected = torch.where(ref_valid.any(-1)[:, None, None], selected, free)
        return {
            "trajectory": trajectory,
            "logits": logits,
            "prediction": prediction,
            "free": free,
            "selected": selected,
            "hidden": self.projection(memory[:, 0]),
        }
