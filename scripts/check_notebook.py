"""Verify notebook source, empty output, attachment and regeneration integrity."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_notebook(root):
    root = Path(root)
    asset_dir = root / "notebooks/assets/architecture"
    manifest = json.loads((asset_dir / "manifest.json").read_text())
    notebook_path = root / "notebooks/pluto_e2e_walkthrough.ipynb"
    notebook = json.loads(notebook_path.read_text())
    for name, expected in manifest["sources"].items():
        if _digest((root / name).read_bytes()) != expected:
            raise ValueError(f"notebook source is stale: {name}")
    cells = notebook["cells"]
    ids = [cell["id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate notebook cell id")
    attachments = {}
    for cell in cells:
        if cell["cell_type"] == "code" and (
            cell["execution_count"] is not None or cell["outputs"]
        ):
            raise ValueError("notebook has stored execution output")
        for name, content in cell.get("attachments", {}).items():
            if name in attachments:
                raise ValueError("duplicate attachment")
            if f"attachment:{name}" not in "".join(cell["source"]):
                raise ValueError("attachment missing from cell")
            attachments[name] = (cell["id"], content["image/svg+xml"].encode())
    if len(manifest["figures"]) != len(attachments):
        raise ValueError("figure inventory size mismatch")
    for item in manifest["figures"]:
        name = item["attachment"]
        anchor, data = attachments[name]
        if (
            anchor != item["id"]
            or _digest(data) != item["sha256"]
            or (asset_dir / name).read_bytes() != data
        ):
            raise ValueError(f"figure integrity mismatch: {name}")
    if {p.name for p in asset_dir.glob("*.svg")} != set(attachments):
        raise ValueError("orphaned or missing SVG asset")
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp)
        for name in manifest["sources"]:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target / name)
        subprocess.run(
            [sys.executable, str(target / "scripts/make_notebook.py")],
            check=True,
            capture_output=True,
        )
        if (
            notebook_path.read_bytes()
            != (target / "notebooks/pluto_e2e_walkthrough.ipynb").read_bytes()
        ):
            raise ValueError("notebook regeneration is not deterministic")
        if (asset_dir / "manifest.json").read_bytes() != (
            target / "notebooks/assets/architecture/manifest.json"
        ).read_bytes():
            raise ValueError("figure manifest regeneration is not deterministic")


if __name__ == "__main__":
    verify_notebook(Path(__file__).resolve().parents[1])
    print("Notebook source, assets, outputs and deterministic regeneration verified.")
