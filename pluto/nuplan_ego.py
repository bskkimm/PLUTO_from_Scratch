"""Read-only ego-only nuPlan SQLite diagnostic; not a complete PLUTO adapter."""

from bisect import bisect_left
import math
from pathlib import Path
import sqlite3
import torch
from .data import agent_history_features


def _yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def read_ego_only(
    path, center_timestamp=None, history_steps=21, future_steps=80, interval_us=100000
):
    """Sample ego poses around a timestamp in microseconds, with 50ms tolerance.

    Inputs are a nuPlan log SQLite path, the current timestamp, historical
    count including the current frame, and future count. Output follows the
    local model's batch contract. Agents, map, route and reference lines are
    absent, so inference necessarily uses the free head. Ego body-frame
    velocities are rotated into the current ego frame. This diagnostic is not
    a substitute for the official builder or closed-loop evaluation.
    """
    if history_steps < 2 or future_steps < 1 or interval_us < 1:
        raise ValueError("invalid sampling configuration")
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        if center_timestamp is None:
            first = connection.execute(
                "SELECT MIN(timestamp) FROM ego_pose"
            ).fetchone()[0]
            if first is None:
                raise ValueError("ego_pose table is empty")
            center_timestamp = first + (history_steps - 1) * interval_us
        tolerance = interval_us // 2
        start = center_timestamp - (history_steps - 1) * interval_us - tolerance
        stop = center_timestamp + future_steps * interval_us + tolerance
        rows = connection.execute(
            "SELECT timestamp,x,y,qw,qx,qy,qz,vx,vy,acceleration_x FROM ego_pose "
            "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start, stop),
        ).fetchall()
    if not rows:
        raise ValueError("no ego poses near requested timestamp")
    timestamps = [row[0] for row in rows]
    selected = []
    for i in range(-(history_steps - 1), future_steps + 1):
        timestamp = center_timestamp + i * interval_us
        idx = bisect_left(timestamps, timestamp)
        options = [j for j in (idx - 1, idx) if 0 <= j < len(rows)]
        closest = min(options, key=lambda j: abs(timestamps[j] - timestamp))
        if abs(timestamps[closest] - timestamp) > tolerance:
            raise ValueError(f"no ego sample close to timestamp {timestamp}")
        selected.append(rows[closest])
    center = selected[history_steps - 1]
    theta0 = _yaw(*center[3:7])
    c0, s0 = math.cos(theta0), math.sin(theta0)
    positions, headings, velocities = [], [], []
    for row in selected:
        dx, dy = row[1] - center[1], row[2] - center[2]
        yaw = _yaw(*row[3:7])
        angle = math.atan2(math.sin(yaw - theta0), math.cos(yaw - theta0))
        c, s = math.cos(angle), math.sin(angle)
        positions.append([c0 * dx + s0 * dy, -s0 * dx + c0 * dy])
        headings.append(angle)
        velocities.append([c * row[7] - s * row[8], s * row[7] + c * row[8]])
    position = torch.tensor(positions, dtype=torch.float32)[None, None]
    heading = torch.tensor(headings, dtype=torch.float32)[None, None]
    velocity = torch.tensor(velocities, dtype=torch.float32)[None, None]
    h, t = history_steps, future_steps
    valid = torch.ones(1, 1, h + t, dtype=torch.bool)
    shape = torch.tensor([2.297, 5.176]).view(1, 1, 1, 2).expand(1, 1, h, 2)
    features, feature_mask = agent_history_features(
        position[:, :, :h],
        velocity[:, :, :h],
        heading[:, :, :h],
        shape,
        valid[:, :, :h],
    )
    target = torch.cat(
        (
            position[:, 0, h:],
            heading[:, 0, h:, None].cos(),
            heading[:, 0, h:, None].sin(),
            velocity[:, 0, h:],
        ),
        -1,
    )
    return {
        "agents": features,
        "agent_mask": feature_mask,
        "ego_state": torch.tensor([[0.0, 0.0, 0.0, center[7], center[9], 0.0]]),
        "agent_pose": torch.zeros(1, 1, 3),
        "map": torch.zeros(1, 0, 1, 10),
        "map_mask": torch.zeros(1, 0, 1, dtype=torch.bool),
        "static": torch.zeros(1, 0, 5),
        "static_mask": torch.zeros(1, 0, dtype=torch.bool),
        "reference": torch.zeros(1, 0, 1, 6),
        "reference_mask": torch.zeros(1, 0, 1, dtype=torch.bool),
        "target": target,
        "target_mask": torch.ones(1, t, dtype=torch.bool),
        "agent_target": torch.zeros(1, 0, t, 6),
        "agent_target_mask": torch.zeros(1, 0, t, dtype=torch.bool),
        "mode_label": torch.zeros(1, dtype=torch.long),
    }
