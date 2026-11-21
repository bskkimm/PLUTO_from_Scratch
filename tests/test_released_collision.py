import math
import pytest
import torch
from pluto.losses import released_sdf_loss
from pluto.model import Pluto, PlutoConfig
from pluto.objective import contrastive_objective, supervised_objective
from pluto.synthetic import sample_batch
from pluto.train import run_training


def test_released_constant_field_normalizes_only_active_circles():
    radius = math.sqrt(2.297**2 + ((4.049 + 1.127) / 3) ** 2) / 2 - 0.2
    field = torch.full((1, 101, 101), 0.5)
    trajectory = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
    value = released_sdf_loss(trajectory, field)
    assert value.item() == pytest.approx((radius - 0.5) * 6 / (6 + 1e-6))
    masked = released_sdf_loss(trajectory, field, valid=torch.tensor([[True, False]]))
    assert masked.item() == pytest.approx((radius - 0.5) * 3 / (3 + 1e-6))


def test_released_offroad_center_outside_and_empty_mask_give_finite_zero():
    trajectory = torch.tensor([[[100.0, 100.0, 1.0, 0.0]]], requires_grad=True)
    field = torch.ones(1, 101, 101)
    assert released_sdf_loss(trajectory, field) == 0
    local = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], requires_grad=True)
    field[:, 50, 50] = -0.1
    assert released_sdf_loss(local, field) == 0
    field[:, 50, 50] = 1
    value = released_sdf_loss(local, field, valid=torch.zeros(1, 1, dtype=torch.bool))
    value.backward()
    assert value == 0 and torch.isfinite(local.grad).all()


def test_released_sampling_gradient_has_expected_direction():
    field = torch.linspace(0.1, 1.0, 101).view(1, 1, 101).expand(1, 101, 101).clone()
    trajectory = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], requires_grad=True)
    value = released_sdf_loss(trajectory, field)
    value.backward()
    assert value > 0 and trajectory.grad[0, 0, 0] < 0
    assert torch.isfinite(trajectory.grad).all()


def test_released_uses_fp32_geometry_for_bfloat16_trajectory():
    field = torch.linspace(0.1, 1.0, 101).view(1, 1, 101).expand(1, 101, 101).clone()
    pose = torch.tensor([[[0.125, 0.0, 1.0, 0.0]]])
    reference = released_sdf_loss(pose, field)
    reduced = released_sdf_loss(pose.bfloat16(), field)
    torch.testing.assert_close(reduced, reference, atol=1e-7, rtol=1e-6)


def test_released_collision_integrates_with_both_objectives():
    torch.manual_seed(41)
    cfg = PlutoConfig(
        dim=16, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4, dropout=0.0
    )
    model = Pluto(cfg).eval()
    batch = sample_batch(2, 4)
    field = torch.full((2, 101, 101), 0.3)
    supervised = supervised_objective(
        model(batch), batch, sdf=field, collision_mode="released"
    )
    contrastive = contrastive_objective(
        model, batch, batch, batch, sdf=field, collision_mode="released"
    )
    assert supervised["auxiliary"] > 0
    torch.testing.assert_close(supervised["auxiliary"], contrastive["auxiliary"])
    contrastive["total"].backward()
    assert torch.isfinite(model.trajectory_head[0].weight.grad).all()


def test_released_mode_requires_sdf_and_rejects_unknown_modes(tmp_path):
    with pytest.raises(ValueError, match="requires an sdf"):
        run_training(tmp_path / "no-sdf", epochs=1, collision_mode="released")
    with pytest.raises(ValueError, match="collision_mode"):
        run_training(tmp_path / "unknown", epochs=1, collision_mode="unknown")


def test_released_collision_validates_shapes_and_resolution():
    trajectory = torch.zeros(1, 2, 4)
    field = torch.ones(1, 7, 7)
    with pytest.raises(ValueError, match="resolution"):
        released_sdf_loss(trajectory, field, resolution=float("nan"))
    with pytest.raises(ValueError, match="valid"):
        released_sdf_loss(trajectory, field, valid=torch.ones(1, 2))
