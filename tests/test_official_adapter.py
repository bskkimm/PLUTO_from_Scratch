import copy
import math
import pytest
import torch
from pluto.losses import imitation_loss
from pluto.model import Pluto, PlutoConfig
from pluto.official_adapter import convert_normalized_batch


def normalized_fixture(future_steps=2):
    b, a, h, t, p = 1, 2, 3, future_steps, 3
    agent_pos = torch.zeros(b, a, h + t, 2)
    agent_pos[..., 0] = torch.arange(h + t)
    agent_pos[:, 1, :, 1] = 3.0
    return {
        "current_state": torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.2, 0.1, 0.3]]),
        "agent": {
            "position": agent_pos,
            "velocity": torch.ones(b, a, h + t, 2),
            "heading": torch.zeros(b, a, h + t),
            "shape": torch.ones(b, a, h + t, 2),
            "valid_mask": torch.ones(b, a, h + t, dtype=torch.bool),
        },
        "map": {
            "point_position": torch.zeros(b, 1, 3, p, 2),
            "point_vector": torch.ones(b, 1, 3, p, 2),
            "point_orientation": torch.zeros(b, 1, 3, p),
            "polygon_center": torch.zeros(b, 1, 3),
            "valid_mask": torch.ones(b, 1, p, dtype=torch.bool),
        },
        "static_objects": {
            "position": torch.zeros(b, 1, 2),
            "heading": torch.zeros(b, 1),
            "shape": torch.ones(b, 1, 2),
            "valid_mask": torch.ones(b, 1, dtype=torch.bool),
        },
        "reference_line": {
            "position": torch.tensor([[[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]]]),
            "vector": torch.ones(b, 1, p, 2),
            "orientation": torch.zeros(b, 1, p),
            "valid_mask": torch.ones(b, 1, p, dtype=torch.bool),
            "future_projection": torch.tensor([[[[30.0, 0.0]] * 8]]),
        },
    }


def test_convert_normalized_official_feature_contract():
    original = normalized_fixture()
    out = convert_normalized_batch(
        original, history_steps=3, future_steps=2, modes=12, radius=120.0
    )
    assert out["agents"].shape == (1, 2, 2, 9)
    assert out["map"].shape == (1, 1, 3, 10)
    assert out["reference"].shape == (1, 1, 3, 6)
    assert torch.equal(out["ego_state"], original["current_state"][:, :6])
    assert torch.equal(out["target"][0, :, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(out["agent_target"][0, 0, :, 0], torch.tensor([1.0, 2.0]))
    assert out["mode_label"].item() == 3


def test_target_frames_keep_ego_axes_and_subtract_each_current_agent_pose():
    original = normalized_fixture()
    original["agent"]["position"][:, 1, :, 1] = torch.arange(5.0)
    original["agent"]["heading"][:, 1] = math.pi / 2
    original["agent"]["heading"][:, 1, 3:] += math.pi / 2
    original["agent"]["velocity"][:, 1] = torch.tensor([2.0, 3.0])
    out = convert_normalized_batch(original, history_steps=3, future_steps=2)
    torch.testing.assert_close(
        out["agent_target"][0, 0],
        torch.tensor([[1.0, 1.0, 0.0, 1.0, 2.0, 3.0], [2.0, 2.0, 0.0, 1.0, 2.0, 3.0]]),
        atol=1e-6,
        rtol=0,
    )


def test_projected_mode_labels_skip_padded_references_and_use_valid_horizon():
    original = normalized_fixture(future_steps=80)
    ref = original["reference_line"]
    for key in ref:
        ref[key] = ref[key].expand(1, 3, *ref[key].shape[2:]).clone()
    ref["valid_mask"][:, 0] = False
    ref["future_projection"].zero_()
    ref["future_projection"][:, 0, :, 1] = 0.0  # Closest but padded.
    ref["future_projection"][:, 1, :, 1] = 2.0
    ref["future_projection"][:, 2, :, 1] = 3.0
    ref["future_projection"][:, 1, 2, 0] = 45.0
    ref["future_projection"][:, 2, 7] = torch.tensor([500.0, 1.0])
    original["agent"]["valid_mask"][:, 0, 3 + 20 :] = False
    out = convert_normalized_batch(original, history_steps=3, future_steps=80)
    assert out["mode_label"].item() == 12 + 4
    original["agent"]["valid_mask"][:, 0] = True
    out = convert_normalized_batch(original, history_steps=3, future_steps=80)
    assert out["mode_label"].item() == 24 + 11


def test_history_only_inference_has_no_labels_and_ignores_future_data():
    original = normalized_fixture()
    expected = convert_normalized_batch(
        original, history_steps=3, future_steps=2, include_targets=False
    )
    for key in ("position", "velocity", "heading", "shape", "valid_mask"):
        original["agent"][key] = original["agent"][key][:, :, :3].clone()
    del original["reference_line"]["future_projection"]
    before = copy.deepcopy(original)
    out = convert_normalized_batch(original, history_steps=3, include_targets=False)
    assert (
        not {"target", "target_mask", "agent_target", "agent_target_mask", "mode_label"}
        & out.keys()
    )
    for key in expected:
        torch.testing.assert_close(out[key], expected[key])
    for category in original:
        if isinstance(original[category], dict):
            for key in original[category]:
                assert torch.equal(original[category][key], before[category][key])
        else:
            assert torch.equal(original[category], before[category])
    model = Pluto(
        PlutoConfig(dim=16, encoder_layers=1, decoder_layers=1, future_steps=2)
    ).eval()
    with torch.no_grad():
        assert torch.isfinite(model(out)["trajectory"]).all()
    with pytest.raises(ValueError, match="history/future"):
        convert_normalized_batch(original, history_steps=3, future_steps=2)


@pytest.mark.parametrize("references", ["empty", "padded"])
def test_empty_scene_labels_and_history_masks_have_finite_backward(references):
    original = normalized_fixture()
    original["agent"]["valid_mask"][:, 1] = False
    original["agent"]["valid_mask"][:, 0, 3:] = False
    original["map"]["valid_mask"][:] = False
    original["static_objects"]["valid_mask"][:] = False
    if references == "empty":
        for key in original["reference_line"]:
            original["reference_line"][key] = original["reference_line"][key][:, :0]
    else:
        original["reference_line"]["valid_mask"][:] = False
    batch = convert_normalized_batch(original, history_steps=3, future_steps=2)
    assert not batch["agent_mask"][:, 1].any()
    assert not batch["agents"][:, 1].any()
    model = Pluto(
        PlutoConfig(dim=16, encoder_layers=1, decoder_layers=1, future_steps=2)
    )
    loss = imitation_loss(model(batch), batch)["total"]
    assert loss.item() == 0.0
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
