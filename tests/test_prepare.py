import hashlib
import json
import pytest
import torch
from pluto.prepare import write_batch_manifest
from pluto.synthetic import sample_batch
from pluto.dataset import BatchManifest


def test_prepare_writes_manifest_with_matching_hash_and_does_not_overwrite(tmp_path):
    batch = sample_batch(1, 4)
    manifest = write_batch_manifest(
        batch, tmp_path, split="mini-diagnostic", provenance={"source": "fixture"}
    )
    raw = json.loads(manifest.read_text())
    data = tmp_path / raw["batches"][0]["path"]
    assert raw["batches"][0]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    loaded = BatchManifest(manifest)[0]
    assert torch.equal(loaded["target"], batch["target"])
    with pytest.raises(FileExistsError):
        write_batch_manifest(
            batch, tmp_path, split="mini-diagnostic", provenance={"source": "fixture"}
        )


def test_prepare_failure_does_not_leave_partial_provenance(tmp_path):
    with pytest.raises(TypeError):
        write_batch_manifest(
            sample_batch(1, 4), tmp_path, split="mini", provenance={"not_json": {1, 2}}
        )
    assert not (tmp_path / "batch.pt").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_training_rejects_prepared_mode_count_mismatch_before_writing_checkpoint(
    tmp_path,
):
    from pluto.train import run_training

    manifest = write_batch_manifest(
        sample_batch(1, 4),
        tmp_path / "data",
        split="mini",
        provenance={"kind": "nuplan-heuristic-single-window-v1", "modes": 12},
    )
    with pytest.raises(ValueError, match="mode"):
        run_training(tmp_path / "run", epochs=1, manifest=manifest, recipe="smoke")
    assert not (tmp_path / "run" / "checkpoint.pt").exists()
