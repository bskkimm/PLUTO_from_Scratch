"""Contrastive imitation objective (paper Eq. 14) for prepared triplets."""

import torch
from .losses import contrastive_loss, imitation_loss, released_sdf_loss, sdf_loss


def _supervised_trajectory(out, batch):
    trajectory = out["trajectory"]
    valid = batch["target_mask"]
    if trajectory is None:
        return out["free"], valid
    b, r, m, t, channels = trajectory.shape
    labels = batch["mode_label"].long()
    ref_valid = batch["reference_mask"].any(-1)
    eligible = ref_valid.any(-1) & valid.any(-1)
    # imitation_loss checks label validity; fallback rows are not supervised here.
    chosen = trajectory.reshape(b, r * m, t, channels)[
        torch.arange(b, device=labels.device), labels.clamp(0, r * m - 1)
    ]
    return chosen, valid & eligible[:, None]


def _collision_loss(trajectory, sdf, resolution, valid, collision_mode):
    if collision_mode == "educational":
        return sdf_loss(trajectory, sdf, resolution=resolution, valid=valid)
    if collision_mode == "released":
        return released_sdf_loss(trajectory, sdf, resolution=resolution, valid=valid)
    raise ValueError("collision_mode must be educational or released")


def supervised_objective(
    out, batch, sdf=None, resolution=0.2, collision_mode="educational"
):
    """Imitation, agent prediction, and an optional labeled-plan SDF penalty."""
    losses = imitation_loss(out, batch)
    auxiliary = losses["total"] * 0
    if sdf is not None:
        trajectory, valid = _supervised_trajectory(out, batch)
        auxiliary = _collision_loss(trajectory, sdf, resolution, valid, collision_mode)
    return {**losses, "auxiliary": auxiliary, "total": losses["total"] + auxiliary}


def contrastive_objective(
    model,
    original,
    positive,
    negative,
    sdf=None,
    positive_sdf=None,
    resolution=0.2,
    temperature=0.1,
    contrast_valid=None,
    collision_mode="educational",
):
    """Supervise original/positive; negative contributes only to Eq. 13.

    Views are prepared externally: augmentations must preserve the positive
    target and invalidate the negative target for the actual scene. This
    function does not assert that a proposed augmentation is causally valid.
    SDF uses positive values for safe space; masks exclude invalid target steps.
    """
    anchor, augmented = model(original), model(positive)
    negative_memory, _ = model.encode_scene(negative)
    negative_hidden = model.projection(negative_memory[:, 0])
    orig_loss, pos_loss = (
        imitation_loss(anchor, original),
        imitation_loss(augmented, positive),
    )
    supervised = (orig_loss["total"] + pos_loss["total"]) * 0.5
    contrast = contrastive_loss(
        anchor["hidden"],
        augmented["hidden"],
        negative_hidden,
        temperature=temperature,
        valid=contrast_valid,
    )
    auxiliary = supervised * 0
    if sdf is not None:
        traj, mask = _supervised_trajectory(anchor, original)
        plus_traj, plus_mask = _supervised_trajectory(augmented, positive)
        plus_sdf = sdf if positive_sdf is None else positive_sdf
        auxiliary = (
            _collision_loss(traj, sdf, resolution, mask, collision_mode)
            + _collision_loss(
                plus_traj, plus_sdf, resolution, plus_mask, collision_mode
            )
        ) * 0.5
    return {
        "total": supervised + contrast + auxiliary,
        "supervised": supervised,
        "contrastive": contrast,
        "auxiliary": auxiliary,
    }
