"""Safe tensor primitives for constructing externally validated contrastive views."""

import torch


def mask_agents(batch, drop):
    """Return a new view with selected non-ego agents removed from input and targets.

    `drop` is [B,A] with True for dropped agents; ego is index 0 and cannot
    be removed. The caller determines interaction/leading status from the scene;
    this primitive does not claim that the resulting view is causally valid.
    """
    if drop.shape != batch["agent_mask"].shape[:2] or drop.dtype != torch.bool:
        raise ValueError("drop must be bool [B,A]")
    if drop[:, 0].any():
        raise ValueError("cannot drop the ego agent")
    view = {k: v.clone() for k, v in batch.items()}
    view["agent_mask"] &= ~drop.unsqueeze(-1)
    view["agent_target_mask"] &= ~drop[:, 1:].unsqueeze(-1)
    return view


def perturb_state(batch, noise):
    """Return a view with caller-supplied state perturbation and unchanged labels.

    This tensor primitive does not rotate maps, filter collisions, or validate
    whether noise preserves the expert trajectory. Use only validated views in
    real-scene training.
    """
    if noise.shape != batch["ego_state"].shape:
        raise ValueError("noise must match ego_state")
    view = {k: v.clone() for k, v in batch.items()}
    view["ego_state"] = view["ego_state"] + noise
    return view


def make_causal_views(
    batch, interactive, leading, traffic_mask=None, traffic_replacement=None
):
    """Build views from externally established interaction and traffic labels.

    For each example, positive removes noninteractive/nonleading agents;
    negative removes interactive/leading agents. If none exist, it may replace
    explicitly selected traffic-light states. A False validity flag means no
    supported negative change occurred. Interaction labels, leading masks and
    appropriate replacement states must come from scene-aware preprocessing;
    this function does not infer or validate causal interactions.
    """
    present = batch["agent_mask"].any(-1)
    if interactive.shape != present.shape or leading.shape != present.shape:
        raise ValueError("interaction and leading masks must be [B,A]")
    if interactive.dtype != torch.bool or leading.dtype != torch.bool:
        raise ValueError("interaction and leading masks must be boolean")
    causal = (interactive | leading) & present
    causal = causal.clone()
    causal[:, 0] = False
    positive_drop = present & ~causal
    positive_drop[:, 0] = False
    positive = mask_agents(batch, positive_drop)
    negative = mask_agents(batch, causal)
    valid = causal.any(-1)
    if traffic_mask is not None or traffic_replacement is not None:
        if (
            traffic_mask is None
            or traffic_replacement is None
            or "traffic_light" not in batch
        ):
            raise ValueError(
                "traffic mask, replacement and original states are all required"
            )
        if (
            traffic_mask.shape != batch["traffic_light"].shape
            or traffic_replacement.shape != traffic_mask.shape
            or traffic_mask.dtype != torch.bool
        ):
            raise ValueError(
                "traffic mask and replacement must match traffic_light [B,P]"
            )
        map_valid = batch["map_mask"].any(-1)
        change = (
            traffic_mask
            & map_valid
            & ~valid[:, None]
            & (traffic_replacement != batch["traffic_light"])
        )
        negative["traffic_light"] = torch.where(
            change, traffic_replacement, batch["traffic_light"]
        )
        valid = valid | change.any(-1)
    return positive, negative, valid
