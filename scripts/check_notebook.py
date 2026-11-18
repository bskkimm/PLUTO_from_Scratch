"""Verify notebook source, empty output, attachment and regeneration integrity."""

import hashlib
import json
import re
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
    notebook_path = root / "notebooks/pluto_e2e_walkthrough_offline.ipynb"
    notebook = json.loads(notebook_path.read_text())
    web_path = root / "notebooks/pluto_e2e_walkthrough.ipynb"
    web_notebook = json.loads(web_path.read_text())
    for name, expected in manifest["sources"].items():
        if _digest((root / name).read_bytes()) != expected:
            raise ValueError(f"notebook source is stale: {name}")
    cells = notebook["cells"]
    if len(web_notebook["cells"]) != len(cells):
        raise ValueError("web and offline notebook cell counts differ")
    figures = {item["id"]: item for item in manifest["figures"]}
    for offline, web in zip(cells, web_notebook["cells"]):
        expected = dict(offline)
        expected.pop("attachments", None)
        if offline["id"] in figures:
            item = figures[offline["id"]]
            expected["source"] = [
                line.replace(f"attachment:{item['attachment']}", item["web_image"])
                for line in offline["source"]
            ]
            references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", "".join(web["source"]))
            if references != [item["web_image"]]:
                raise ValueError("web notebook image reference is missing or invalid")
            path = web_path.parent / item["web_image"]
            if not path.is_file() or path.suffix != ".png":
                raise ValueError("web notebook image is missing or not PNG")
        if expected != web:
            raise ValueError("web and offline notebook content differs")
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
    previews = asset_dir / "previews"
    preview_manifest = json.loads((previews / "preview-manifest.json").read_text())
    if set(preview_manifest["figures"]) != set(attachments):
        raise ValueError("preview manifest inventory mismatch")
    expected_pngs = set()
    for name, entry in preview_manifest["figures"].items():
        preview_name = Path(name).with_suffix(".png").name
        expected_pngs.add(preview_name)
        if (
            _digest((asset_dir / name).read_bytes()) != entry["svg_sha256"]
            or _digest((previews / preview_name).read_bytes()) != entry["png_sha256"]
        ):
            raise ValueError(f"preview integrity mismatch: {name}")
    if {p.name for p in previews.glob("*.png")} != expected_pngs:
        raise ValueError("orphaned or missing preview")
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
            != (target / notebook_path.relative_to(root)).read_bytes()
            or web_path.read_bytes()
            != (target / web_path.relative_to(root)).read_bytes()
        ):
            raise ValueError("notebook regeneration is not deterministic")
        if (asset_dir / "manifest.json").read_bytes() != (
            target / "notebooks/assets/architecture/manifest.json"
        ).read_bytes():
            raise ValueError("figure manifest regeneration is not deterministic")


if __name__ == "__main__":
    verify_notebook(Path(__file__).resolve().parents[1])
    print("Notebook source, assets, outputs and deterministic regeneration verified.")
