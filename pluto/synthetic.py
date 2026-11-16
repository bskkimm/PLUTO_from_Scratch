"""Deterministic synthetic input fixture, not a nuPlan adapter."""
import torch


def sample_batch(batch_size=2, future_steps=4, seed=42):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g)
    b, a, r = batch_size, 3, 2
    target = rand(b, future_steps, 6) * 0.1
    return {'agents': rand(b, a, 5, 9), 'agent_mask': torch.ones(b, a, 5, dtype=torch.bool),
            'ego_state': rand(b, 6), 'map': rand(b, 3, 5, 6), 'map_mask': torch.ones(b, 3, 5, dtype=torch.bool),
            'static': rand(b, 2, 5), 'static_mask': torch.ones(b, 2, dtype=torch.bool),
            'reference': rand(b, r, 6, 6), 'reference_mask': torch.ones(b, r, 6, dtype=torch.bool),
            'target': target, 'target_mask': torch.ones(b, future_steps, dtype=torch.bool),
            'agent_target': rand(b, a - 1, future_steps, 6) * 0.1,
            'agent_target_mask': torch.ones(b, a - 1, future_steps, dtype=torch.bool),
            'mode_label': torch.zeros(b, dtype=torch.long)}
