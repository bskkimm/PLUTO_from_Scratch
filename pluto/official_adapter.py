"""Convert already-normalized official PLUTO feature dictionaries to local tensors.

This adapter does not extract raw nuPlan scenes, install nuPlan, or recreate the
released model. Keys follow jchengai/pluto revision b9964b6. Input is batched,
CPU or GPU torch tensors and must already be normalized by the official builder.
"""

import torch
from .data import agent_history_features


def convert_normalized_batch(
    data, history_steps=21, future_steps=80, modes=12, radius=120.0
):
    if history_steps < 2 or future_steps < 1 or modes < 1 or radius <= 0:
        raise ValueError("invalid history, future, mode count or radius")
    agent, maps, static, ref = (
        data[k] for k in ("agent", "map", "static_objects", "reference_line")
    )
    h, t = history_steps, future_steps
    if agent["position"].shape[2] < h + t:
        raise ValueError("agent history/future is shorter than requested")
    agents, agent_mask = agent_history_features(
        agent["position"][:, :, :h],
        agent["velocity"][:, :, :h],
        agent["heading"][:, :, :h],
        agent["shape"][:, :, :h],
        agent["valid_mask"][:, :, :h],
    )
    pose = torch.cat(
        (agent["position"][:, :, h - 1], agent["heading"][:, :, h - 1, None]), -1
    )
    centers = maps["point_position"][:, :, 0]
    map_vectors = maps["point_vector"][:, :, 0]
    map_yaw = maps["point_orientation"][:, :, 0]
    map_feat = torch.cat(
        (
            centers - maps["polygon_center"][..., None, :2],
            map_vectors,
            torch.stack((map_yaw.cos(), map_yaw.sin()), -1),
            maps["point_position"][:, :, 1] - centers,
            maps["point_position"][:, :, 2] - centers,
        ),
        -1,
    )
    rpos, rvec, ryaw = ref["position"], ref["vector"], ref["orientation"]
    reference = torch.cat(
        (rpos - rpos[..., :1, :], rvec, torch.stack((ryaw.cos(), ryaw.sin()), -1)), -1
    )
    ref_pose = torch.cat((rpos[:, :, 0], ryaw[:, :, 0, None]), -1)
    static_pose = torch.cat((static["position"], static["heading"].unsqueeze(-1)), -1)
    static_features = torch.cat((static_pose, static["shape"]), -1)
    future_pos = agent["position"][:, :, h : h + t] - agent["position"][:, :, h - 1 : h]
    future_yaw = agent["heading"][:, :, h : h + t] - agent["heading"][:, :, h - 1 : h]
    target = torch.cat(
        (
            future_pos,
            torch.stack((future_yaw.cos(), future_yaw.sin()), -1),
            agent["velocity"][:, :, h : h + t],
        ),
        -1,
    )
    future_valid = agent["valid_mask"][:, :, h : h + t]
    bs, r = rpos.shape[:2]
    labels = torch.zeros(bs, dtype=torch.long, device=rpos.device)
    if r:
        valid_ref = ref["valid_mask"].any(-1)
        projections = ref["future_projection"]
        endpoint_index = (
            (future_valid[:, 0].sum(-1) // 10)
            .long()
            .clamp(max=projections.shape[2] - 1)
        )
        projected = projections[torch.arange(bs, device=rpos.device), :, endpoint_index]
        ref_index = projected[..., 1].masked_fill(~valid_ref, float("inf")).argmin(-1)
        distance = projected[torch.arange(bs, device=rpos.device), ref_index, 0]
        mode = (distance / (radius / modes)).long().clamp(0, modes - 1)
        labels = ref_index * modes + mode
    return {
        "agents": agents,
        "agent_mask": agent_mask,
        "agent_pose": pose,
        "ego_state": data["current_state"][:, :6],
        "map": map_feat,
        "map_mask": maps["valid_mask"],
        "map_pose": maps["polygon_center"],
        "agent_type": agent.get(
            "category",
            torch.zeros(
                bs, agent["position"].shape[1], dtype=torch.long, device=rpos.device
            ),
        ),
        "map_type": maps.get(
            "polygon_type",
            torch.zeros(bs, centers.shape[1], dtype=torch.long, device=rpos.device),
        ),
        "on_route": maps.get(
            "polygon_on_route",
            torch.zeros(bs, centers.shape[1], dtype=torch.long, device=rpos.device),
        ),
        "traffic_light": maps.get(
            "polygon_tl_status",
            torch.zeros(bs, centers.shape[1], dtype=torch.long, device=rpos.device),
        ),
        "speed_limit": maps.get(
            "polygon_speed_limit", centers.new_zeros(bs, centers.shape[1])
        ),
        "speed_known": maps.get(
            "polygon_has_speed_limit",
            torch.zeros(bs, centers.shape[1], dtype=torch.bool, device=rpos.device),
        ),
        "static_type": static.get(
            "category",
            torch.zeros(bs, static_pose.shape[1], dtype=torch.long, device=rpos.device),
        ),
        "static": static_features,
        "static_pose": static_pose,
        "static_mask": static["valid_mask"],
        "reference": reference,
        "reference_mask": ref["valid_mask"],
        "reference_pose": ref_pose,
        "target": target[:, 0],
        "target_mask": future_valid[:, 0],
        "agent_target": target[:, 1:],
        "agent_target_mask": future_valid[:, 1:],
        "mode_label": labels,
    }
