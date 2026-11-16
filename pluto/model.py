"""Vector scene encoder and factorized lateral/longitudinal planner.

All spatial inputs are in the ego frame, in metres. Masks are True for valid
entries. The first agent is ego; other agent future predictions are agent-relative.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class PlutoConfig:
    dim: int = 128
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    modes: int = 12
    future_steps: int = 80
    dropout: float = 0.1


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class PointEncoder(nn.Module):
    """Masked pointwise MLP and max pooling; empty polylines map to zero."""
    def __init__(self, channels, dim):
        super().__init__()
        self.net = mlp(channels, dim, dim)

    def forward(self, x, mask):
        y = self.net(x).masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pooled = y.max(dim=-2).values
        return torch.where(mask.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))


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
        self.lateral = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.longitudinal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, memory, ref_valid, scene_valid):
        b, r, m, d = q.shape
        x = q.transpose(1, 2).reshape(b * m, r, d)
        # An all-masked row yields NaNs in attention; a temporary dummy is ignored later.
        valid = ref_valid.clone()
        valid[~valid.any(-1), 0] = True
        mask = (~valid)[:, None, :].expand(b, m, r).reshape(b * m, r)
        y = self.norms[0](x)
        x = x + self.drop(self.lateral(y, y, y, key_padding_mask=mask, need_weights=False)[0])
        x = x.reshape(b, m, r, d).transpose(1, 2).reshape(b * r, m, d)
        y = self.norms[1](x)
        x = x + self.drop(self.longitudinal(y, y, y, need_weights=False)[0])
        x = x.reshape(b, r * m, d)
        y = self.norms[2](x)
        x = x + self.drop(self.cross(y, memory, memory, key_padding_mask=~scene_valid, need_weights=False)[0])
        x = x + self.drop(self.ff(self.norms[3](x)))
        return x.reshape(b, r, m, d)


class Pluto(nn.Module):
    """PLUTO-style model. See docs/paper-to-code.md for deliberate differences."""
    def __init__(self, config=PlutoConfig()):
        super().__init__()
        self.config = config
        d = config.dim
        self.agent_encoder = PointEncoder(9, d)
        self.map_encoder = PointEncoder(6, d)
        self.static_encoder = mlp(5, d, d)
        self.ego_encoder = mlp(6, d, d)
        self.type_embedding = nn.Embedding(3, d)
        self.scene_position = FourierPosition(d)
        self.reference_position = FourierPosition(d)
        layer = nn.TransformerEncoderLayer(d, config.heads, d * 4, config.dropout, batch_first=True, norm_first=True)
        self.scene_encoder = nn.TransformerEncoder(layer, config.encoder_layers, norm=nn.LayerNorm(d))
        self.reference_encoder = PointEncoder(6, d)
        self.mode = nn.Parameter(torch.randn(config.modes, d) * 0.01)
        self.query_projection = nn.Linear(2 * d, d)
        self.decoder = nn.ModuleList([DecoderLayer(d, config.heads, config.dropout) for _ in range(config.decoder_layers)])
        self.trajectory_head = mlp(d, 2 * d, config.future_steps * 6)
        self.score_head = mlp(d, d, 1)
        self.prediction_head = mlp(d, 2 * d, config.future_steps * 6)
        self.free_head = mlp(d, 2 * d, config.future_steps * 6)
        self.projection = mlp(d, d, d)

    def encode_scene(self, batch):
        """Encode scene tokens; contrastive negatives need no trajectory decoder."""
        agents, maps, static = batch['agents'], batch['map'], batch['static']
        b, a = agents.shape[:2]
        agent_valid = batch['agent_mask'].any(-1)
        # The ego state supplies a valid scene token even if its history is absent.
        agent_valid = agent_valid.clone()
        agent_valid[:, 0] = True
        agent = self.agent_encoder(agents, batch['agent_mask'])
        agent = torch.cat([self.ego_encoder(batch['ego_state']).unsqueeze(1), agent[:, 1:]], dim=1)
        tokens = torch.cat([agent + self.type_embedding.weight[0],
                            self.map_encoder(maps, batch['map_mask']) + self.type_embedding.weight[1],
                            self.static_encoder(static) + self.type_embedding.weight[2]], dim=1)
        scene_valid = torch.cat([agent_valid, batch['map_mask'].any(-1), batch['static_mask']], dim=1)
        agent_pose = batch.get('agent_pose', agents.new_zeros(b, a, 3))
        map_pose = batch.get('map_pose', maps.new_zeros(b, maps.shape[1], 3))
        static_pose = batch.get('static_pose', static[..., :3])
        pose = torch.cat((agent_pose, map_pose, static_pose), dim=1)
        tokens = tokens + self.scene_position(pose)
        memory = self.scene_encoder(tokens, src_key_padding_mask=~scene_valid)
        return memory, scene_valid

    def forward(self, batch):
        c = self.config
        b, a = batch['agents'].shape[:2]
        memory, scene_valid = self.encode_scene(batch)
        prediction = self.prediction_head(memory[:, 1:a]).reshape(b, a - 1, c.future_steps, 6)
        free = self.free_head(memory[:, 0]).reshape(b, c.future_steps, 6)
        refs, ref_mask = batch['reference'], batch['reference_mask']
        r = refs.shape[1]
        if r == 0:
            return {'trajectory': None, 'logits': None, 'prediction': prediction, 'free': free,
                    'selected': free, 'hidden': self.projection(memory[:, 0])}
        ref_valid = ref_mask.any(-1)
        ref_pose = batch.get('reference_pose', refs.new_zeros(b, r, 3))
        ref = self.reference_encoder(refs, ref_mask) + self.reference_position(ref_pose)
        q = self.query_projection(torch.cat([ref.unsqueeze(2).expand(b, r, c.modes, c.dim),
                                             self.mode[None, None].expand(b, r, c.modes, c.dim)], -1))
        for layer in self.decoder:
            q = layer(q, memory, ref_valid, scene_valid)
        trajectory = self.trajectory_head(q).reshape(b, r, c.modes, c.future_steps, 6)
        logits = self.score_head(q).squeeze(-1).masked_fill(~ref_valid.unsqueeze(-1), -1e9)
        best = logits.reshape(b, -1).argmax(-1)
        selected = trajectory.reshape(b, r * c.modes, c.future_steps, 6)[torch.arange(b, device=best.device), best]
        selected = torch.where(ref_valid.any(-1)[:, None, None], selected, free)
        return {'trajectory': trajectory, 'logits': logits, 'prediction': prediction, 'free': free,
                'selected': selected, 'hidden': self.projection(memory[:, 0])}
