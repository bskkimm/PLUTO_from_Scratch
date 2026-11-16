"""Explicit manifest for preprocessed tensor batches; no dataset SDK required."""
import hashlib
import io
import json
from pathlib import Path
import re
import torch


class BatchManifest:
    """Load immutable, locally prepared batches with per-file SHA-256 checks.

    Format is `pluto-batches-v1`; `split` identifies the dataset split and
    `batches` is a nonempty list of path/sha256 objects. Paths are relative to
    the manifest directory. Each file contains a dict of batched CPU tensors
    with the model's documented keys. This is not a nuPlan feature builder.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw)
        if obj.get('format') != 'pluto-batches-v1' or not isinstance(obj.get('split'), str):
            raise ValueError('invalid manifest format or split')
        self.split = obj['split']
        entries = obj.get('batches')
        if not isinstance(entries, list) or not entries:
            raise ValueError('manifest must contain batches')
        self.entries = []
        for item in entries:
            candidate = (self.path.parent / item['path']).resolve()
            if not candidate.is_relative_to(self.path.parent):
                raise ValueError('batch path outside manifest directory')
            digest = item['sha256']
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('invalid SHA-256')
            self.entries.append((candidate, digest))
        self.verify()

    def __len__(self):
        return len(self.entries)

    def _bytes(self, i):
        path, expected = self.entries[i]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'batch checksum mismatch: {path}')
        return data

    def verify(self):
        for i in range(len(self)):
            self._bytes(i)

    def __getitem__(self, i):
        batch = torch.load(io.BytesIO(self._bytes(i)), map_location='cpu', weights_only=True)
        if not isinstance(batch, dict) or not all(isinstance(v, torch.Tensor) for v in batch.values()):
            raise ValueError('batch must be a dictionary of tensors')
        required = {'agents', 'agent_mask', 'ego_state', 'map', 'map_mask', 'static', 'static_mask',
                    'reference', 'reference_mask', 'target', 'target_mask', 'agent_target',
                    'agent_target_mask', 'mode_label'}
        if not required <= batch.keys():
            raise ValueError(f'missing batch keys: {sorted(required - batch.keys())}')
        return batch
