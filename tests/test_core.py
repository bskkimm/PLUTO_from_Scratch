import torch
from pluto.model import Pluto, PlutoConfig
from pluto.losses import masked_smooth_l1, contrastive_loss, sdf_loss, imitation_loss
from pluto.synthetic import sample_batch


def test_model_shapes_masks_and_fallback():
    torch.manual_seed(1)
    cfg = PlutoConfig(
        dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(batch_size=2, future_steps=4)
    batch["reference_mask"][1] = False
    out = model(batch)
    assert out["trajectory"].shape == (2, 2, 3, 4, 6)
    assert out["prediction"].shape == (2, 2, 4, 6)
    assert torch.isfinite(out["trajectory"]).all()
    assert torch.equal(out["selected"][1], out["free"][1])


def test_empty_mask_is_finite_and_zero():
    x = torch.randn(2, 3, 6, requires_grad=True)
    loss = masked_smooth_l1(x, torch.zeros_like(x), torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(x.grad).all()


def test_contrastive_prefers_similar_positive():
    anchor = torch.tensor([[1.0, 0.0]])
    assert contrastive_loss(anchor, anchor, -anchor) < contrastive_loss(
        anchor, -anchor, anchor
    )


def test_sdf_interpolation_is_differentiable():
    sdf = torch.arange(25.0).reshape(1, 5, 5)
    traj = torch.tensor([[[0.2, 0.0, 1.0, 0.0]]], requires_grad=True)
    loss = sdf_loss(traj, sdf, resolution=1.0, offsets=(0.0,), radius=20.0)
    loss.backward()
    assert loss.item() > 0 and traj.grad[0, 0, 0].abs() > 0


def test_optimization_step():
    torch.manual_seed(2)
    cfg = PlutoConfig(
        dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4
    )
    model = Pluto(cfg)
    batch = sample_batch(batch_size=2, future_steps=4)
    out = model(batch)
    losses = imitation_loss(out, batch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    losses["total"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    before = model.trajectory_head[0].weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.trajectory_head[0].weight)


def test_teacher_forcing_projects_endpoint_to_nearest_line_and_mode():
    from pluto.losses import assign_modes

    refs = torch.tensor(
        [[[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], [[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]]]]
    )
    masks = torch.ones(1, 2, 3, dtype=torch.bool)
    endpoint = torch.tensor([[7.0, 4.0]])
    assert assign_modes(endpoint, refs, masks, modes=3).item() == 4
    assert assign_modes(torch.tensor([[15.0, 4.0]]), refs, masks, modes=3).item() == 5


def test_zero_reference_count_uses_free_head():
    cfg = PlutoConfig(
        dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch["reference"] = batch["reference"][:, :0]
    batch["reference_mask"] = batch["reference_mask"][:, :0]
    output = model(batch)
    assert output["trajectory"] is None
    assert torch.equal(output["selected"], output["free"])
    assert torch.isfinite(imitation_loss(output, batch)["total"])


def test_padded_reference_values_do_not_change_selection():
    torch.manual_seed(9)
    cfg = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch["reference_mask"][:, 1] = False
    with torch.no_grad():
        first = model(batch)["selected"]
        batch["reference"][:, 1] = 10000.0
        second = model(batch)["selected"]
    assert torch.allclose(first, second)


def test_scene_position_changes_valid_actor_encoding_but_not_padding():
    torch.manual_seed(5)
    cfg = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch["agent_pose"] = torch.zeros(2, 3, 3)
    with torch.no_grad():
        before = model(batch)["selected"]
        batch["agent_pose"][:, 1, 0] = 20.0
        after = model(batch)["selected"]
        assert not torch.allclose(before, after)
        batch["agent_mask"][:, 1] = False
        masked = model(batch)["selected"]
        batch["agent_pose"][:, 1, 0] = -100.0
        masked_changed = model(batch)["selected"]
        assert torch.allclose(masked, masked_changed)


def test_out_of_range_mode_labels_are_rejected():
    import pytest

    cfg = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    batch = sample_batch(2, 4)
    batch["mode_label"][0] = 999
    with pytest.raises(ValueError, match="mode_label"):
        imitation_loss(Pluto(cfg)(batch), batch)


def test_state_attention_dropout_masks_only_optional_channels():
    from pluto.model import StateEncoder

    torch.manual_seed(11)
    state = StateEncoder(dim=16, heads=4, dropout=1.0)
    x = torch.zeros(2, 6)
    altered = x.clone()
    altered[:, 3:] = 50.0
    state.train()
    assert torch.allclose(state(x), state(altered))
    state.eval()
    assert not torch.allclose(state(x), state(altered))


def test_map_attributes_change_valid_scene_output():
    torch.manual_seed(12)
    cfg = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch["traffic_light"] = torch.zeros(2, 3, dtype=torch.long)
    with torch.no_grad():
        original = model(batch)["selected"]
        batch["traffic_light"][:, 0] = 1
        changed = model(batch)["selected"]
    assert not torch.allclose(original, changed)


def test_local_history_attention_cannot_read_outside_its_window():
    from pluto.model import LocalAttentionBlock

    torch.manual_seed(13)
    block = LocalAttentionBlock(dim=8, heads=2, window=3).eval()
    history = torch.randn(1, 7, 8)
    valid = torch.ones(1, 7, dtype=torch.bool)
    with torch.no_grad():
        original = block(history, valid)
        changed = history.clone()
        changed[:, 5] += 100.0
        revised = block(changed, valid)
    assert torch.allclose(original[:, 0], revised[:, 0])
    assert not torch.allclose(original[:, 5], revised[:, 5])


def test_history_encoder_ignores_padding_and_handles_empty_agents():
    from pluto.model import HistoryEncoder

    torch.manual_seed(14)
    encoder = HistoryEncoder(channels=9, dim=16, heads=4).eval()
    features = torch.randn(2, 2, 5, 9)
    mask = torch.tensor(
        [
            [[True, True, False, False, False], [False] * 5],
            [[True] * 5, [True, False, True, False, False]],
        ]
    )
    with torch.no_grad():
        first = encoder(features, mask)
        changed = features.clone()
        changed[~mask] = 10000.0
        second = encoder(changed, mask)
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert torch.equal(first[0, 1], torch.zeros_like(first[0, 1]))


def test_sdf_figure_hinge_example():
    trajectory = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    distance_field = torch.full((1, 5, 5), 0.7)
    actual = sdf_loss(
        trajectory,
        distance_field,
        resolution=1.0,
        offsets=(0.0,),
        radius=1.0,
        margin=0.2,
    )
    assert torch.allclose(actual, torch.tensor(0.5), atol=1e-6)


def test_inference_figure_ade_example():
    prediction = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert torch.equal((prediction - target).norm(dim=-1).mean(), torch.tensor(0.5))


def test_half_precision_masked_references_remain_finite():
    torch.manual_seed(31)
    cfg = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    batch["reference_mask"][:] = False
    with torch.autocast("cpu", dtype=torch.float16):
        out = model(batch)
        loss = imitation_loss(out, batch)["total"]
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(loss)
