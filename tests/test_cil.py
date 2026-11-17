import torch
from pluto.data import agent_history_features
from pluto.model import Pluto, PlutoConfig
from pluto.objective import contrastive_objective
from pluto.synthetic import sample_batch


def test_history_vectors_use_consecutive_valid_pairs():
    position = torch.tensor([[[[0.0, 0.0], [2.0, 1.0], [3.0, 3.0]]]])
    velocity = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]]])
    heading = torch.tensor([[[0.0, 0.0, 1.0]]])
    shape = torch.ones(1, 1, 3, 2)
    valid = torch.tensor([[[True, True, False]]])
    features, pair_mask = agent_history_features(
        position, velocity, heading, shape, valid
    )
    assert features.shape == (1, 1, 2, 9)
    assert torch.equal(pair_mask, torch.tensor([[[True, False]]]))
    assert torch.equal(features[0, 0, 0, :4], torch.tensor([2.0, 1.0, 1.0, 0.0]))
    assert torch.count_nonzero(features[0, 0, 1]) == 0


def test_cil_only_uses_negative_as_contrastive_input():
    torch.manual_seed(1)
    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(config).eval()
    original = sample_batch(2, 4, seed=1)
    positive = sample_batch(2, 4, seed=2)
    negative = sample_batch(2, 4, seed=3)
    positive["target"] = original["target"].clone()
    positive["agent_target"] = original["agent_target"].clone()
    first = contrastive_objective(model, original, positive, negative)
    negative["target"].fill_(9999.0)
    negative["agent_target"].fill_(9999.0)
    second = contrastive_objective(model, original, positive, negative)
    assert torch.allclose(first["total"], second["total"])
    first["total"].backward()
    assert model.projection[0].weight.grad.abs().sum() > 0
    assert model.trajectory_head[0].weight.grad.abs().sum() > 0


def test_cil_auxiliary_loss_changes_total_and_backpropagates():
    torch.manual_seed(2)
    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(config).eval()
    a, p, n = (sample_batch(2, 4, seed=i) for i in range(3))
    loss = contrastive_objective(
        model, a, p, n, sdf=torch.zeros(2, 11, 11), resolution=1.0
    )
    assert loss["auxiliary"] > 0
    loss["total"].backward()
    assert torch.isfinite(model.trajectory_head[0].weight.grad).all()


def test_mask_agents_keeps_original_and_drops_only_requested():
    from pluto.augmentation import mask_agents

    scene = sample_batch(2, 4)
    drop = torch.tensor([[False, True, False], [False, False, True]])
    view = mask_agents(scene, drop)
    assert scene["agent_mask"].all()
    assert not view["agent_mask"][0, 1].any()
    assert view["agent_mask"][0, 2].all()
    assert not view["agent_target_mask"][0, 0].any()
    assert view["agent_target_mask"][0, 1].all()
    try:
        mask_agents(scene, torch.ones(2, 3, dtype=torch.bool))
    except ValueError as exc:
        assert "ego" in str(exc)
    else:
        raise AssertionError("must reject dropping ego")


def test_sdf_ignores_invalid_steps():
    from pluto.losses import sdf_loss

    trajectory = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    sdf = torch.zeros(1, 7, 7)
    mask = torch.tensor([[True, False]])
    one = sdf_loss(trajectory, sdf, offsets=(0.0,), valid=mask)
    both = sdf_loss(trajectory, sdf, offsets=(0.0,))
    assert torch.equal(one, both)
    assert sdf_loss(trajectory, sdf, offsets=(0.0,), valid=torch.zeros_like(mask)) == 0


def test_causal_views_drop_only_declared_agents_and_mark_missing_negative():
    from pluto.augmentation import make_causal_views

    batch = sample_batch(2, 4)
    interactive = torch.tensor([[False, True, False], [False, False, False]])
    leading = torch.tensor([[False, False, False], [False, False, False]])
    positive, negative, valid = make_causal_views(batch, interactive, leading)
    assert valid.tolist() == [True, False]
    assert positive["agent_mask"][0, 1].all() and not positive["agent_mask"][0, 2].any()
    assert not negative["agent_mask"][0, 1].any() and negative["agent_mask"][0, 2].all()
    assert batch["agent_mask"].all()


def test_causal_views_invert_only_explicitly_supplied_lights():
    from pluto.augmentation import make_causal_views

    batch = sample_batch(2, 4)
    batch["traffic_light"] = torch.tensor([[1, 1, 2], [1, 1, 2]])
    interaction = torch.zeros(2, 3, dtype=torch.bool)
    lights = torch.tensor([[True, False, False], [False, False, False]])
    replacement = torch.tensor([[2, 2, 2], [2, 2, 2]])
    _, neg, valid = make_causal_views(
        batch, interaction, interaction, lights, replacement
    )
    assert valid.tolist() == [True, False]
    assert neg["traffic_light"].tolist() == [[2, 1, 2], [1, 1, 2]]
    assert batch["traffic_light"][0, 0] == 1


def test_noncontrastive_training_can_include_sdf_auxiliary():
    from pluto.objective import supervised_objective

    torch.manual_seed(23)
    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(config).eval()
    batch = sample_batch(2, 4)
    output = model(batch)
    without = supervised_objective(output, batch)
    with_sdf = supervised_objective(
        output, batch, sdf=torch.zeros(2, 11, 11), resolution=1.0
    )
    assert with_sdf["auxiliary"] > 0
    assert with_sdf["total"] > without["total"]
    with_sdf["total"].backward()
    assert torch.isfinite(model.trajectory_head[0].weight.grad).all()


def test_insert_leading_agent_uses_empty_slot_and_preserves_original():
    from pluto.augmentation import insert_agent

    batch = sample_batch(2, 4)
    batch["agent_mask"][0, 2] = False
    batch["agent_target_mask"][0, 1] = False
    batch["agent_pose"] = torch.zeros(2, 3, 3)
    batch["agent_type"] = torch.tensor([[0, 1, 0], [0, 2, 1]])
    donor = torch.tensor([1, 1])
    donor_agent = torch.tensor([1, 2])
    slots = torch.tensor([2, -1])
    poses = torch.tensor([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    view, changed = insert_agent(batch, donor, donor_agent, slots, poses)
    assert changed.tolist() == [True, False]
    assert view["agent_mask"][0, 2].all() and not batch["agent_mask"][0, 2].any()
    assert torch.equal(view["agents"][0, 2], batch["agents"][1, 1])
    assert torch.equal(view["agent_pose"][0, 2], poses[0])
    assert view["agent_type"][0, 2] == 2
    assert not view["agent_target_mask"][0, 1].any()


def test_insert_agent_rejects_occupied_ego_and_invalid_donor():
    from pluto.augmentation import insert_agent
    import pytest

    batch = sample_batch(1, 4)
    batch["agent_pose"] = torch.zeros(1, 3, 3)
    args = (batch, torch.tensor([0]), torch.tensor([1]))
    with pytest.raises(ValueError, match="empty non-ego"):
        insert_agent(*args, torch.tensor([1]), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="empty non-ego"):
        insert_agent(*args, torch.tensor([0]), torch.zeros(1, 3))


def test_causal_views_can_use_validated_insertion_when_no_interaction():
    from pluto.augmentation import make_causal_views

    batch = sample_batch(2, 4)
    batch["agent_mask"][0, 2] = False
    batch["agent_pose"] = torch.zeros(2, 3, 3)
    empty = torch.zeros(2, 3, dtype=torch.bool)
    insertion = {
        "donor_batch": torch.tensor([1, 1]),
        "donor_agent": torch.tensor([1, 1]),
        "destination_slot": torch.tensor([2, -1]),
        "poses": torch.tensor([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    _, negative, valid = make_causal_views(batch, empty, empty, insertion=insertion)
    assert valid.tolist() == [True, False]
    assert negative["agent_mask"][0, 2].any()
    assert not negative["agent_target_mask"][0, 1].any()


def test_insertion_uses_original_donor_when_donor_is_removed_in_its_negative_view():
    from pluto.augmentation import make_causal_views

    batch = sample_batch(2, 4)
    batch["agent_mask"][0, 2] = False
    batch["agent_pose"] = torch.zeros(2, 3, 3)
    interaction = torch.tensor([[False, False, False], [False, True, False]])
    insertion = {
        "donor_batch": torch.tensor([1, 1]),
        "donor_agent": torch.tensor([1, 1]),
        "destination_slot": torch.tensor([2, -1]),
        "poses": torch.zeros(2, 3),
    }
    _, negative, valid = make_causal_views(
        batch, interaction, torch.zeros_like(interaction), insertion=insertion
    )
    assert valid.tolist() == [True, True]
    assert negative["agent_mask"][0, 2].all()
    assert not negative["agent_mask"][1, 1].any()
