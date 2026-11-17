import pytest
import torch
from pluto.costmap import signed_distance_field


def test_signed_distance_has_expected_sign_and_resolution():
    drivable = torch.tensor(
        [[[False, False, False], [False, True, False], [False, False, False]]]
    )
    obstacle = torch.zeros_like(drivable)
    field = signed_distance_field(drivable, obstacle, resolution=0.2)
    assert torch.allclose(field[0, 1, 1], torch.tensor(0.2))
    assert torch.allclose(field[0, 1, 0], torch.tensor(-0.2))
    obstacle[0, 1, 1] = True
    blocked = signed_distance_field(drivable, obstacle, resolution=0.2)
    assert (blocked < 0).all() and torch.isfinite(blocked).all()


def test_distance_map_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        signed_distance_field(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        signed_distance_field(
            torch.zeros(2, 2, dtype=torch.bool), torch.zeros(3, 3, dtype=torch.bool)
        )
