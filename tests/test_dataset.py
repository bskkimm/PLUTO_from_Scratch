import hashlib
import json
import torch
import pytest
from pluto.dataset import BatchManifest
from pluto.synthetic import sample_batch
from pluto.train import run_training


def make_manifest(tmp_path):
    batch_path = tmp_path / "batch.pt"
    torch.save(sample_batch(2, 4), batch_path)
    digest = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "pluto-batches-v1",
                "split": "train",
                "batches": [{"path": "batch.pt", "sha256": digest}],
            }
        )
    )
    return manifest_path, batch_path


def test_manifest_detects_modified_batch_before_loading(tmp_path):
    manifest, data = make_manifest(tmp_path)
    loader = BatchManifest(manifest)
    assert loader[0]["target"].shape == (2, 4, 6)
    data.write_bytes(data.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        loader[0]


def test_manifest_rejects_path_escape(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format": "pluto-batches-v1",
                "split": "train",
                "batches": [{"path": "../bad.pt", "sha256": "0" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="outside"):
        BatchManifest(path)


def test_resume_rejects_changed_manifest_before_checkpoint_write(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    out = tmp_path / "run"
    run_training(out, epochs=2, stop_after=1, manifest=manifest)
    old = (out / "checkpoint.pt").read_bytes()
    obj = json.loads(manifest.read_text())
    obj["split"] = "changed"
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="config"):
        run_training(out, epochs=2, resume=True, manifest=manifest)
    assert (out / "checkpoint.pt").read_bytes() == old


def test_manifest_evaluation_requires_matching_manifest(tmp_path):
    from pluto.evaluate import evaluate_checkpoint

    manifest, _ = make_manifest(tmp_path)
    checkpoint = run_training(tmp_path / "run", epochs=1, manifest=manifest)
    with pytest.raises(ValueError, match="manifest"):
        evaluate_checkpoint(checkpoint)
    result = evaluate_checkpoint(checkpoint, manifest=manifest)
    assert result["examples"] == 2
    assert result["ade_m"] >= 0


def test_manifest_cil_uses_explicit_causal_metadata_and_resumes(tmp_path):
    batch = sample_batch(2, 4)
    batch["interactive"] = torch.tensor([[False, True, False], [False, False, False]])
    batch["leading"] = torch.zeros(2, 3, dtype=torch.bool)
    batch["sdf"] = torch.ones(2, 9, 9)
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "pluto-batches-v1",
                "split": "train",
                "batches": [
                    {
                        "path": "batch.pt",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    out = tmp_path / "run"
    run_training(out, epochs=2, stop_after=1, manifest=manifest, cil=True)
    run_training(out, epochs=2, manifest=manifest, cil=True, resume=True)
    checkpoint = torch.load(
        out / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 2


def test_gradient_accumulation_updates_partial_final_group(tmp_path):
    batch = sample_batch(2, 4)
    entries = []
    for i in range(3):
        path = tmp_path / f"batch-{i}.pt"
        torch.save(batch, path)
        entries.append(
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps({"format": "pluto-batches-v1", "split": "train", "batches": entries})
    )
    checkpoint = run_training(
        tmp_path / "run", epochs=1, manifest=manifest, accumulation_steps=2
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["optimizer_steps"] == 2
    assert state["config"]["accumulation_steps"] == 2
