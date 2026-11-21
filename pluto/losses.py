"""Planning, agent prediction, contrastive and differentiable SDF objectives."""

import math
import torch
import torch.nn.functional as F


def masked_smooth_l1(pred, target, valid):
    errors = F.smooth_l1_loss(pred, target, reduction="none").sum(-1)
    return (errors * valid.to(errors.dtype)).sum() / valid.sum().clamp(min=1)


def contrastive_loss(anchor, positive, negative, temperature=0.1, valid=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    a, p, n = (F.normalize(x, dim=-1) for x in (anchor, positive, negative))
    logits = torch.stack([(a * p).sum(-1), (a * n).sum(-1)], -1) / temperature
    loss = F.cross_entropy(
        logits,
        torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
        reduction="none",
    )
    if valid is None:
        return loss.mean() if loss.numel() else logits.sum() * 0
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def sdf_loss(
    trajectory,
    sdf,
    resolution=0.2,
    offsets=(-1.0, 0.0, 1.0),
    radius=1.0,
    margin=0.0,
    valid=None,
):
    """Eq. 12: average sum of hinge penalties per timestep; x right, y up.

    SDF is positive in safe space, shape [B,H,W], with its origin at the
    center pixel. Out-of-bounds positions get zero distance (conservative).
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    h, w = sdf.shape[-2:]
    if h < 2 or w < 2:
        raise ValueError("SDF dimensions must be at least 2")
    off = trajectory.new_tensor(offsets).view(1, 1, -1, 1)
    centers = trajectory[..., None, :2] + off * trajectory[..., None, 2:4]
    gx = 2 * (centers[..., 0] / resolution + (w - 1) / 2) / (w - 1) - 1
    gy = 2 * (-centers[..., 1] / resolution + (h - 1) / 2) / (h - 1) - 1
    grid = torch.stack((gx, gy), -1)
    distance = F.grid_sample(
        sdf.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)
    cost = (radius + margin - distance).clamp(min=0).sum(-1)
    if valid is None:
        return cost.mean()
    return (cost * valid.to(cost.dtype)).sum() / valid.sum().clamp(min=1)


def released_sdf_loss(trajectory, sdf, resolution=0.2, valid=None):
    """Released PLUTO three-circle ESDF loss, with an optional timestep mask.

    Trajectories contain x, y, cos(yaw), sin(yaw), in metres and the current
    ego frame. The ESDF is positive on drivable space, [B,H,W], x right/y up.
    The released operator ignores future validity; passing ``valid`` extends
    it to exclude unsupervised timesteps, including rows with no labeled plan.
    With ``valid=None`` it follows the released operator's numerical conventions.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] < 4:
        raise ValueError("trajectory must be [B,T,C] with C >= 4")
    if sdf.ndim != 3 or sdf.shape[0] != trajectory.shape[0] or min(sdf.shape[-2:]) < 2:
        raise ValueError("sdf must be [B,H,W] with matching B and H,W >= 2")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    if valid is not None and (
        valid.shape != trajectory.shape[:2] or valid.dtype != torch.bool
    ):
        raise ValueError("valid must be boolean [B,T]")
    h, w = sdf.shape[-2:]
    count = 3
    width, front, rear = 2.297, 4.049, 1.127
    interval = (front + rear) / count
    radius = math.sqrt(width**2 + interval**2) / 2 - resolution
    # The released operator constructs offsets in float32, including under AMP.
    offsets = torch.tensor(
        [-rear + interval * (i + 0.5) for i in range(count)],
        dtype=torch.float32,
        device=trajectory.device,
    ).view(1, 1, count, 1)
    centers = trajectory[..., None, :2] + offsets * trajectory[..., None, 2:4]
    origin = torch.tensor([w // 2, h // 2], device=trajectory.device)
    grid = torch.stack((centers[..., 0], -centers[..., 1]), -1) / resolution / origin
    inside = ((grid < 0.95) & (grid > -0.95)).all(-1)
    on_road = sdf[:, h // 2, w // 2] > 0
    distance = F.grid_sample(
        sdf.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)
    cost = radius - distance
    active = inside & on_road[:, None, None] & (cost > 0)
    if valid is not None:
        active = active & valid[..., None]
    return cost.masked_fill(~active, 0).sum() / (active.sum() + 1e-6)


def imitation_loss(out, batch):
    target, valid = batch["target"], batch["target_mask"]
    free = masked_smooth_l1(out["free"], target, valid)
    pred = masked_smooth_l1(
        out["prediction"], batch["agent_target"], batch["agent_target_mask"]
    )
    if out["trajectory"] is None:
        zero = free * 0
        return {
            "total": free + pred,
            "regression": zero,
            "classification": zero,
            "free": free,
            "prediction": pred,
        }
    traj, logits = out["trajectory"], out["logits"]
    b, r, m, t, _ = traj.shape
    ref_valid = batch["reference_mask"].any(-1)
    eligible = ref_valid.any(-1) & valid.any(-1)
    labels = batch["mode_label"].long()
    if (eligible & ((labels < 0) | (labels >= r * m))).any():
        raise ValueError("mode_label is outside the candidate range")
    idx = labels.clamp(0, r * m - 1)
    selected_valid = ref_valid.gather(1, (idx // m).unsqueeze(1)).squeeze(1)
    if (eligible & ~selected_valid).any():
        raise ValueError("mode_label refers to a masked reference line")
    chosen = traj.reshape(b, r * m, t, 6)[torch.arange(b, device=idx.device), idx]
    reg = masked_smooth_l1(chosen, target, valid & eligible[:, None])
    cls = (
        F.cross_entropy(logits.reshape(b, -1)[eligible], idx[eligible])
        if eligible.any()
        else logits.sum() * 0
    )
    return {
        "total": reg + cls + free + pred,
        "regression": reg,
        "classification": cls,
        "free": free,
        "prediction": pred,
    }


def assign_modes(endpoint, reference_xy, reference_mask, modes):
    """Project [B,2] endpoints onto valid polylines, returning flattened R*M labels.

    Longitudinal bins divide each reference length into M-1 equal intervals;
    the last bin is for projections at or beyond the end. Points are in metres
    in the same coordinate frame. Every sample needs a non-degenerate line.
    """
    if modes < 2 or reference_xy.shape[-2] < 2:
        raise ValueError("need at least two modes and two points per reference")
    start, end = reference_xy[..., :-1, :], reference_xy[..., 1:, :]
    delta = end - start
    length2 = (delta * delta).sum(-1)
    valid = reference_mask[..., :-1] & reference_mask[..., 1:] & (length2 > 1e-12)
    if not valid.any(dim=(-1, -2)).all():
        raise ValueError("every sample needs a valid, non-degenerate reference segment")
    length = length2.sqrt()
    offset = endpoint[:, None, None, :] - start
    u = ((offset * delta).sum(-1) / length2.clamp(min=1e-12)).clamp(0.0, 1.0)
    projected = start + u[..., None] * delta
    distance = (
        ((endpoint[:, None, None, :] - projected) ** 2)
        .sum(-1)
        .masked_fill(~valid, float("inf"))
    )
    b, r, s = distance.shape
    segment_index = distance.reshape(b, -1).argmin(-1)
    ref_index, point_index = segment_index // s, segment_index % s
    lengths = torch.where(valid, length, torch.zeros_like(length))
    accumulated = lengths.cumsum(-1) - lengths
    batch_idx = torch.arange(b, device=endpoint.device)
    arc = (
        accumulated[batch_idx, ref_index, point_index]
        + u[batch_idx, ref_index, point_index]
        * length[batch_idx, ref_index, point_index]
    )
    total = lengths.sum(-1)[batch_idx, ref_index]
    longitudinal = (
        torch.floor(arc / (total / (modes - 1)).clamp(min=1e-12))
        .long()
        .clamp(0, modes - 1)
    )
    return ref_index * modes + longitudinal
