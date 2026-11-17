"""Signed distance fields from externally rasterized drivable/obstacle masks."""

import torch


def signed_distance_field(drivable, obstacle, resolution=0.2):
    """Return positive distances in safe cells and negative outside, in metres.

    Inputs are equally shaped bool tensors [H,W] or [B,H,W], with True for
    drivable/occupied cells. Cell-center Euclidean distances are computed on
    CPU using SciPy; input geometry and rasterization are the caller's
    responsibility. Fully safe/unsafe masks use a finite image-diagonal bound.
    The returned constant tensor is not differentiable with respect to masks.
    """
    if (
        drivable.dtype != torch.bool
        or obstacle.dtype != torch.bool
        or drivable.shape != obstacle.shape
    ):
        raise ValueError(
            "drivable and obstacle must be identically shaped boolean masks"
        )
    if drivable.ndim not in (2, 3) or min(drivable.shape[-2:]) < 2 or resolution <= 0:
        raise ValueError(
            "expected [H,W] or [B,H,W] with spatial dimensions >=2 and positive resolution"
        )
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("cost-map construction requires SciPy") from exc
    safe = (drivable & ~obstacle).detach().cpu().numpy()
    single = safe.ndim == 2
    if single:
        safe = safe[None]
    fields = []
    bound = (safe.shape[-2] ** 2 + safe.shape[-1] ** 2) ** 0.5 * resolution
    for sample in safe:
        if sample.all():
            field = np.full(sample.shape, bound, dtype=np.float32)
        elif not sample.any():
            field = np.full(sample.shape, -bound, dtype=np.float32)
        else:
            field = (
                (distance_transform_edt(sample) - distance_transform_edt(~sample))
                * resolution
            ).astype(np.float32)
        fields.append(field)
    result = torch.from_numpy(np.stack(fields)).to(drivable.device)
    return result[0] if single else result
