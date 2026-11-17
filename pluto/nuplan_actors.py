"""Bounded, read-only nuPlan ego and tracked-object SQLite extraction."""

from bisect import bisect_left, bisect_right
import math
from pathlib import Path
import sqlite3
import torch
from .data import agent_history_features
from .nuplan_ego import _yaw, read_ego_only

_DYNAMIC = {"vehicle": 1, "pedestrian": 2, "bicycle": 3}
_STATIC = {"czone_sign": 0, "barrier": 1, "traffic_cone": 2, "generic_object": 3}


def read_ego_and_actors(
    path,
    center_timestamp,
    history_steps=21,
    future_steps=80,
    interval_us=100000,
    radius=120.0,
    max_agents=48,
    max_static=30,
):
    """Extract actors present now and their nearby history/targets in the ego frame.

    Database timestamps use microseconds. Track selection uses current-frame
    positions at or before the planning timestamp; future boxes are used solely for prediction targets. Raw
    lidar boxes are in the map frame, whereas ego velocities are body-frame.
    This omits lane maps, route, reference lines, causal labels and the official
    devkit's interpolation/filtering; it is a bounded diagnostic adapter.
    """
    if center_timestamp is None or radius <= 0 or max_agents < 1 or max_static < 0:
        raise ValueError(
            "center timestamp, positive radius and valid limits are required"
        )
    result = read_ego_only(
        path, center_timestamp, history_steps, future_steps, interval_us
    )
    path = Path(path).resolve(strict=True)
    tolerance = interval_us // 2
    start = center_timestamp - (history_steps - 1) * interval_us - tolerance
    end = center_timestamp + future_steps * interval_us + tolerance
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        ego = connection.execute(
            "SELECT timestamp,x,y,qw,qx,qy,qz FROM ego_pose WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC LIMIT 1",
            (center_timestamp - tolerance, center_timestamp),
        ).fetchone()
        frames = connection.execute(
            "SELECT token,timestamp FROM lidar_pc WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start, end),
        ).fetchall()
        if ego is None or not frames:
            raise ValueError("no ego or lidar frame near the requested timestamp")
        times = [row[1] for row in frames]
        chosen = []
        for i in range(-(history_steps - 1), future_steps + 1):
            target = center_timestamp + i * interval_us
            ix = bisect_left(times, target)
            if i <= 0:
                closest = bisect_right(times, target) - 1
            else:
                choices = [j for j in (ix - 1, ix) if 0 <= j < len(frames)]
                closest = (
                    min(choices, key=lambda j: abs(times[j] - target))
                    if choices
                    else -1
                )
            chosen.append(
                frames[closest][0]
                if closest >= 0 and abs(times[closest] - target) <= tolerance
                else None
            )
        tokens = list(dict.fromkeys(token for token in chosen if token is not None))
        if chosen[history_steps - 1] is None:
            raise ValueError("no lidar frame near the current timestamp")
        placeholders = ",".join("?" for _ in tokens)
        sql = (
            "SELECT lb.lidar_pc_token,lb.track_token,lb.x,lb.y,lb.yaw,lb.vx,lb.vy,lb.width,lb.length,c.name "
            "FROM lidar_box lb JOIN track t ON lb.track_token=t.token "
            "JOIN category c ON t.category_token=c.token WHERE lb.lidar_pc_token IN ("
            + placeholders
            + ")"
        )
        boxes = connection.execute(sql, tokens).fetchall()
    theta0 = _yaw(*ego[3:7])
    c0, s0 = math.cos(theta0), math.sin(theta0)
    per_frame = {}
    for token, track, x, y, yaw, vx, vy, width, length, category in boxes:
        if any(value is None for value in (x, y, yaw, vx, vy, width, length)):
            continue
        dx, dy = x - ego[1], y - ego[2]
        angle = math.atan2(math.sin(yaw - theta0), math.cos(yaw - theta0))
        value = (
            c0 * dx + s0 * dy,
            -s0 * dx + c0 * dy,
            angle,
            c0 * vx + s0 * vy,
            -s0 * vx + c0 * vy,
            width,
            length,
            category,
        )
        per_frame.setdefault(token, {})[track] = value
    present = per_frame.get(chosen[history_steps - 1], {})
    ordered = sorted(
        (
            (v[0] ** 2 + v[1] ** 2, track, v)
            for track, v in present.items()
            if v[0] ** 2 + v[1] ** 2 <= radius**2
        ),
        key=lambda item: (item[0], item[1]),
    )
    agents = [(track, value) for _, track, value in ordered if value[7] in _DYNAMIC][
        : max_agents - 1
    ]
    statics = [value for _, _, value in ordered if value[7] in _STATIC][:max_static]
    n, steps = len(agents), history_steps + future_steps
    position = torch.zeros(1, n, steps, 2)
    heading = torch.zeros(1, n, steps)
    velocity = torch.zeros(1, n, steps, 2)
    shape = torch.zeros(1, n, steps, 2)
    valid = torch.zeros(1, n, steps, dtype=torch.bool)
    for i, (track, _) in enumerate(agents):
        for t, token in enumerate(chosen):
            value = per_frame.get(token, {}).get(track)
            if value is None:
                continue
            position[0, i, t] = torch.tensor(value[:2])
            heading[0, i, t] = value[2]
            velocity[0, i, t] = torch.tensor(value[3:5])
            shape[0, i, t] = torch.tensor(value[5:7])
            valid[0, i, t] = True
    if n:
        h = history_steps
        features, mask = agent_history_features(
            position[:, :, :h],
            velocity[:, :, :h],
            heading[:, :, :h],
            shape[:, :, :h],
            valid[:, :, :h],
        )
        result["agents"] = torch.cat((result["agents"], features), 1)
        result["agent_mask"] = torch.cat((result["agent_mask"], mask), 1)
        agent_pose = torch.cat((position[:, :, h - 1], heading[:, :, h - 1, None]), -1)
        result["agent_pose"] = torch.cat((result["agent_pose"], agent_pose), 1)
        future_angle = heading[:, :, h:] - heading[:, :, h - 1 : h]
        future_target = torch.cat(
            (
                position[:, :, h:] - position[:, :, h - 1 : h],
                torch.stack((future_angle.cos(), future_angle.sin()), -1),
                velocity[:, :, h:],
            ),
            -1,
        )
        result["agent_target"] = future_target
        result["agent_target_mask"] = valid[:, :, h:]
    result["agent_type"] = torch.tensor(
        [[0] + [_DYNAMIC[value[7]] for _, value in agents]], dtype=torch.long
    )
    if statics:
        result["static"] = torch.tensor(
            [[[v[0], v[1], v[2], v[5], v[6]] for v in statics]], dtype=torch.float32
        )
        result["static_mask"] = torch.ones(1, len(statics), dtype=torch.bool)
        result["static_type"] = torch.tensor(
            [[_STATIC[v[7]] for v in statics]], dtype=torch.long
        )
    return result
