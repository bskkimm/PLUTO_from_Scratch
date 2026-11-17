import torch
from pluto.train import run_training


def test_epoch_resume_matches_uninterrupted(tmp_path):
    whole = tmp_path / "whole"
    split = tmp_path / "split"
    run_training(whole, epochs=3, seed=123)
    run_training(split, epochs=3, seed=123, stop_after=1)
    run_training(split, epochs=3, seed=123, resume=True)
    a = torch.load(whole / "checkpoint.pt", map_location="cpu", weights_only=False)
    b = torch.load(split / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert a["epoch"] == b["epoch"] == 3
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key


def test_resume_rejects_changed_seed(tmp_path):
    run_training(tmp_path, epochs=2, seed=1, stop_after=1)
    try:
        run_training(tmp_path, epochs=2, seed=2, resume=True)
    except ValueError as e:
        assert "config" in str(e)
    else:
        raise AssertionError("expected incompatible config rejection")


def test_tiny_fixture_learns(tmp_path, capsys):
    import json

    run_training(tmp_path, epochs=5, seed=123)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["loss"] < rows[0]["loss"]


def test_cil_training_resume_matches_uninterrupted(tmp_path):
    a, b = tmp_path / "whole", tmp_path / "split"
    run_training(a, epochs=2, seed=17, cil=True)
    run_training(b, epochs=2, seed=17, cil=True, stop_after=1)
    run_training(b, epochs=2, seed=17, cil=True, resume=True)
    x = torch.load(a / "checkpoint.pt", map_location="cpu", weights_only=False)
    y = torch.load(b / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key in x["model"]:
        assert torch.equal(x["model"][key], y["model"][key]), key


def test_cpu_bfloat16_training_and_resume(tmp_path):
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    run_training(whole, epochs=2, precision="bf16")
    run_training(resumed, epochs=2, stop_after=1, precision="bf16")
    run_training(resumed, epochs=2, resume=True, precision="bf16")
    checkpoint = torch.load(
        resumed / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    baseline = torch.load(
        whole / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["config"]["precision"] == "bf16"
    assert "scaler" in checkpoint
    assert all(
        torch.equal(checkpoint["model"][key], baseline["model"][key])
        for key in baseline["model"]
    )


def test_unavailable_cuda_fails_clearly(tmp_path):
    import pytest

    if torch.cuda.is_available():
        pytest.skip("requires no CUDA")
    with pytest.raises(RuntimeError, match="CUDA"):
        run_training(tmp_path, epochs=1, device="cuda")
    assert not (tmp_path / "checkpoint.pt").exists()
