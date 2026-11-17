"""Illustrative rollout scoring utilities, not the nuPlan/PDM evaluator."""

import torch
import torch.nn.functional as F


def score_rollouts(
    rollouts,
    valid,
    obstacles=None,
    obstacle_mask=None,
    collision_radius=2.0,
    sdf=None,
    resolution=0.2,
):
    """Score progress with diagnostic collision and drivable-area gates.

    `rollouts` [B,K,T,4] holds x,y,yaw,speed in ego-frame metres and m/s;
    `valid` [B,K] masks candidate proposals. `obstacles`, when provided, is
    [B,A,2] fixed ego-frame point centers; `obstacle_mask` is [B,A]. `sdf`
    [B,H,W] is positive for drivable space and has a centered ego origin,
    x right and y up. Out-of-map samples count as non-drivable. A score of
    zero is assigned to colliding, off-road, and masked candidates; otherwise
    the score is nonnegative end-to-end displacement in metres.

    This deliberately simple diagnostic does not model box geometry, moving
    actors, traffic rules, TTC, comfort, official metric aggregation, or safety.
    """
    if (
        rollouts.ndim != 4
        or rollouts.shape[-1] != 4
        or valid.shape != rollouts.shape[:2]
    ):
        raise ValueError("expected rollout [B,K,T,4] and valid [B,K]")
    if (
        valid.dtype != torch.bool
        or rollouts.shape[2] == 0
        or collision_radius < 0
        or resolution <= 0
    ):
        raise ValueError("invalid mask, time dimension, radius or resolution")
    b, k, _, _ = rollouts.shape
    collision = torch.zeros(b, k, dtype=torch.bool, device=rollouts.device)
    if obstacles is not None:
        if obstacles.ndim != 3 or obstacles.shape[0] != b or obstacles.shape[-1] != 2:
            raise ValueError("obstacles must be [B,A,2]")
        if obstacle_mask is None:
            obstacle_mask = torch.ones(
                obstacles.shape[:2], dtype=torch.bool, device=obstacles.device
            )
        if (
            obstacle_mask.shape != obstacles.shape[:2]
            or obstacle_mask.dtype != torch.bool
        ):
            raise ValueError("obstacle_mask must be boolean [B,A]")
        distances = (rollouts[..., :2].unsqueeze(-2) - obstacles[:, None, None]).norm(
            dim=-1
        )
        collision = (
            (distances <= collision_radius) & obstacle_mask[:, None, None]
        ).any(dim=(-1, -2))
    drivable = torch.ones_like(collision)
    if sdf is not None:
        if sdf.ndim != 3 or sdf.shape[0] != b or min(sdf.shape[-2:]) < 2:
            raise ValueError("sdf must be [B,H,W] with H,W >= 2")
        height, width = sdf.shape[-2:]
        xy = rollouts[..., :2]
        gx = 2 * xy[..., 0] / (resolution * (width - 1))
        gy = -2 * xy[..., 1] / (resolution * (height - 1))
        grid = torch.stack((gx, gy), -1).reshape(b, k, -1, 2)
        in_bounds = (grid.abs() <= 1).all(-1)
        sampled = F.grid_sample(
            sdf.unsqueeze(1).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        drivable = ((sampled > 0) & in_bounds).all(-1)
    progress = (rollouts[:, :, -1, :2] - rollouts[:, :, 0, :2]).norm(dim=-1)
    score = torch.where(
        valid & ~collision & drivable, progress, torch.zeros_like(progress)
    )
    return score, {"collision": collision, "drivable": drivable, "progress_m": progress}
