"""Safe tensor primitives for constructing externally validated contrastive views."""
import torch


def mask_agents(batch, drop):
    """Return a new view with selected non-ego agents removed from input and targets.

    `drop` is [B,A] with True for dropped agents; ego is index 0 and cannot
    be removed. The caller determines interaction/leading status from the scene;
    this primitive does not claim that the resulting view is causally valid.
    """
    if drop.shape != batch['agent_mask'].shape[:2] or drop.dtype != torch.bool:
        raise ValueError('drop must be bool [B,A]')
    if drop[:, 0].any():
        raise ValueError('cannot drop the ego agent')
    view = {k: v.clone() for k, v in batch.items()}
    view['agent_mask'] &= ~drop.unsqueeze(-1)
    view['agent_target_mask'] &= ~drop[:, 1:].unsqueeze(-1)
    return view


def perturb_state(batch, noise):
    """Return a view with caller-supplied state perturbation and unchanged labels.

    This tensor primitive does not rotate maps, filter collisions, or validate
    whether noise preserves the expert trajectory. Use only validated views in
    real-scene training.
    """
    if noise.shape != batch['ego_state'].shape:
        raise ValueError('noise must match ego_state')
    view = {k: v.clone() for k, v in batch.items()}
    view['ego_state'] = view['ego_state'] + noise
    return view
