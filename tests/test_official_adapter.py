import torch
from pluto.official_adapter import convert_normalized_batch


def test_convert_normalized_official_feature_contract():
    b, a, h, t, p = 1, 2, 3, 2, 3
    agent_pos = torch.tensor([[[[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]], [[0., 3.], [1., 3.], [2., 3.], [3., 3.], [4., 3.]]]])
    original = {'current_state': torch.zeros(b, 6),
        'agent': {'position': agent_pos, 'velocity': torch.ones(b, a, h + t, 2), 'heading': torch.zeros(b, a, h + t), 'shape': torch.ones(b, a, h + t, 2), 'valid_mask': torch.ones(b, a, h + t, dtype=torch.bool)},
        'map': {'point_position': torch.zeros(b, 1, 3, p, 2), 'point_vector': torch.ones(b, 1, 3, p, 2), 'point_orientation': torch.zeros(b, 1, 3, p), 'polygon_center': torch.zeros(b, 1, 3), 'valid_mask': torch.ones(b, 1, p, dtype=torch.bool)},
        'static_objects': {'position': torch.zeros(b, 1, 2), 'heading': torch.zeros(b, 1), 'shape': torch.ones(b, 1, 2), 'valid_mask': torch.ones(b, 1, dtype=torch.bool)},
        'reference_line': {'position': torch.tensor([[[[0., 0.], [10., 0.], [20., 0.]]]]), 'vector': torch.ones(b, 1, p, 2), 'orientation': torch.zeros(b, 1, p), 'valid_mask': torch.ones(b, 1, p, dtype=torch.bool), 'future_projection': torch.tensor([[[[30., 0.]] * 8]])}}
    out = convert_normalized_batch(original, history_steps=h, future_steps=t, modes=12, radius=120.)
    assert out['agents'].shape == (1, 2, 2, 9)
    assert out['map'].shape == (1, 1, 3, 6)
    assert out['reference'].shape == (1, 1, 3, 6)
    assert torch.equal(out['target'][0, :, 0], torch.tensor([1., 2.]))
    assert torch.equal(out['agent_target'][0, 0, :, 0], torch.tensor([1., 2.]))
    assert out['mode_label'].item() == 3
