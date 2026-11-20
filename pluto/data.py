"""Vector preprocessing independent of dataset and simulator SDKs."""

import torch


def agent_history_features(position, velocity, heading, shape, valid):
    """Return official-style 9-channel consecutive history vectors and mask.

    Inputs [B,A,H,2] positions (m), velocities (m/s), dimensions (m),
    headings [B,A,H] (rad), and valid observations [B,A,H]. The output
    channels are dx,dy,dvx,dvy,cos(dheading),sin(dheading),width,length,
    pair-valid. A pair is valid only when both endpoints are valid.
    All inputs must already share the intended coordinate frame.
    """
    pair_valid = valid[..., 1:] & valid[..., :-1]
    dh = heading[..., 1:] - heading[..., :-1]
    features = torch.cat(
        (
            position[..., 1:, :] - position[..., :-1, :],
            velocity[..., 1:, :] - velocity[..., :-1, :],
            torch.stack((dh.cos(), dh.sin()), dim=-1),
            shape[..., 1:, :],
            pair_valid.unsqueeze(-1).to(position.dtype),
        ),
        dim=-1,
    )
    return torch.where(
        pair_valid.unsqueeze(-1), features, torch.zeros_like(features)
    ), pair_valid
