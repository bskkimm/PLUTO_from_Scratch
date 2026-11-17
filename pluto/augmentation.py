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
    batch,
    interactive,
    leading,
    traffic_mask=None,
    traffic_replacement=None,
    insertion=None,
):
    """Build views from externally established interaction and traffic labels.

    For each example, positive removes noninteractive/nonleading agents;
    negative removes interactive/leading agents. If none exist, it may insert
    a caller-validated agent or replace selected traffic-light states. A False
    validity flag means no supported negative change occurred. Interaction labels, leading masks and
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
    if insertion is not None:
        required = {"donor_batch", "donor_agent", "destination_slot", "poses"}
        if set(insertion) != required:
            raise ValueError(
                "insertion must contain donor_batch, donor_agent, destination_slot and poses"
            )
        slots = torch.where(
            valid,
            torch.full_like(insertion["destination_slot"], -1),
            insertion["destination_slot"],
        )
        negative, inserted = insert_agent(
            negative,
            insertion["donor_batch"],
            insertion["donor_agent"],
            slots,
            insertion["poses"],
        )
        valid = valid | inserted
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


def insert_agent(batch, donor_batch, donor_agent, destination_slot, poses):
    """Insert caller-selected donor history into empty non-ego slots.

    `donor_batch`, `donor_agent`, and `destination_slot` are integer [B]
    indices; slot -1 means no insertion for that example. `poses` is [B,3]
    in the recipient ego frame. Donor history is copied unchanged, so caller
    must transform/validate its orientation and select a causally meaningful
    placement. Inserted agent targets are masked: the copied history supplies
    no valid ground truth in the recipient scene. No mutation of input occurs.
    """
    history = batch["agents"]
    b, a = history.shape[:2]
    if "agent_pose" not in batch or poses.shape != (b, 3):
        raise ValueError("agent_pose and replacement poses [B,3] are required")
    for index in (donor_batch, donor_agent, destination_slot):
        if index.shape != (b,) or index.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be integer [B] tensors")
    selected = destination_slot >= 0
    if ((destination_slot < -1) | (destination_slot >= a)).any():
        raise ValueError("destination must be -1 or an empty non-ego slot")
    if (selected & (destination_slot == 0)).any():
        raise ValueError("destination must be an empty non-ego slot")
    if (
        selected
        & (
            (donor_batch < 0)
            | (donor_batch >= b)
            | (donor_agent <= 0)
            | (donor_agent >= a)
        )
    ).any():
        raise ValueError("invalid donor indices")
    rows = selected.nonzero(as_tuple=True)[0]
    dest = destination_slot[rows].long()
    src_batch = donor_batch[rows].long()
    src_agent = donor_agent[rows].long()
    if batch["agent_mask"][rows, dest].any():
        raise ValueError("destination must be an empty non-ego slot")
    if not batch["agent_mask"][src_batch, src_agent].any(dim=-1).all():
        raise ValueError("donor must contain valid history")
    view = {key: value.clone() for key, value in batch.items()}
    view["agents"][rows, dest] = batch["agents"][src_batch, src_agent]
    view["agent_mask"][rows, dest] = batch["agent_mask"][src_batch, src_agent]
    view["agent_pose"][rows, dest] = poses[rows]
    if "agent_type" in batch:
        view["agent_type"][rows, dest] = batch["agent_type"][src_batch, src_agent]
    view["agent_target_mask"][rows, dest - 1] = False
    return view, selected
