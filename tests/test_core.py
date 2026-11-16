import torch
from pluto.model import Pluto, PlutoConfig
from pluto.losses import masked_smooth_l1, contrastive_loss, sdf_loss, imitation_loss
from pluto.synthetic import sample_batch


def test_model_shapes_masks_and_fallback():
    torch.manual_seed(1)
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4)
    model = Pluto(cfg).eval()
    batch = sample_batch(batch_size=2, future_steps=4)
    batch['reference_mask'][1] = False
    out = model(batch)
    assert out['trajectory'].shape == (2, 2, 3, 4, 6)
    assert out['prediction'].shape == (2, 2, 4, 6)
    assert torch.isfinite(out['trajectory']).all()
    assert torch.equal(out['selected'][1], out['free'][1])


def test_empty_mask_is_finite_and_zero():
    x = torch.randn(2, 3, 6, requires_grad=True)
    loss = masked_smooth_l1(x, torch.zeros_like(x), torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(x.grad).all()


def test_contrastive_prefers_similar_positive():
    anchor = torch.tensor([[1., 0.]])
    assert contrastive_loss(anchor, anchor, -anchor) < contrastive_loss(anchor, -anchor, anchor)


def test_sdf_interpolation_is_differentiable():
    sdf = torch.arange(25.).reshape(1, 5, 5)
    traj = torch.tensor([[[0.2, 0., 1., 0.]]], requires_grad=True)
    loss = sdf_loss(traj, sdf, resolution=1., offsets=(0.,), radius=20.)
    loss.backward()
    assert loss.item() > 0 and traj.grad[0, 0, 0].abs() > 0


def test_optimization_step():
    torch.manual_seed(2)
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4)
    model = Pluto(cfg)
    batch = sample_batch(batch_size=2, future_steps=4)
    out = model(batch)
    losses = imitation_loss(out, batch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad(); losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before = model.trajectory_head[0].weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.trajectory_head[0].weight)


def test_teacher_forcing_projects_endpoint_to_nearest_line_and_mode():
    from pluto.losses import assign_modes
    refs = torch.tensor([[[[0., 0.], [5., 0.], [10., 0.]], [[0., 5.], [5., 5.], [10., 5.]]]])
    masks = torch.ones(1, 2, 3, dtype=torch.bool)
    endpoint = torch.tensor([[7., 4.]])
    assert assign_modes(endpoint, refs, masks, modes=3).item() == 4
    assert assign_modes(torch.tensor([[15., 4.]]), refs, masks, modes=3).item() == 5


def test_zero_reference_count_uses_free_head():
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4)
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch['reference'] = batch['reference'][:, :0]
    batch['reference_mask'] = batch['reference_mask'][:, :0]
    output = model(batch)
    assert output['trajectory'] is None
    assert torch.equal(output['selected'], output['free'])
    assert torch.isfinite(imitation_loss(output, batch)['total'])


def test_padded_reference_values_do_not_change_selection():
    torch.manual_seed(9)
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4, dropout=0.)
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch['reference_mask'][:, 1] = False
    with torch.no_grad():
        first = model(batch)['selected']
        batch['reference'][:, 1] = 10000.
        second = model(batch)['selected']
    assert torch.allclose(first, second)
