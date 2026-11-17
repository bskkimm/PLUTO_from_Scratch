import sqlite3
import pytest
import torch
from pluto.nuplan_ego import read_ego_only


def test_read_only_ego_log_converts_to_local_frame(tmp_path):
    db = tmp_path / "mini.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "create table ego_pose (timestamp integer, x real, y real, qw real, qx real, qy real, qz real, vx real, vy real, acceleration_x real)"
    )
    connection.executemany(
        "insert into ego_pose values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (i * 100000, i * 0.1, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            for i in range(10)
        ],
    )
    connection.commit()
    connection.close()
    batch = read_ego_only(db, center_timestamp=300000, history_steps=3, future_steps=4)
    assert torch.allclose(batch["target"][0, :, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.allclose(batch["target"][0, :, 1], torch.zeros(4))
    assert batch["agents"].shape == (1, 1, 2, 9)
    assert batch["reference"].shape[1] == 0
    with pytest.raises(ValueError, match="timestamp"):
        read_ego_only(db, center_timestamp=0, history_steps=3, future_steps=4)


def test_ego_coordinates_rotate_global_path_into_heading_frame(tmp_path):
    import math

    db = tmp_path / "rotated.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "create table ego_pose (timestamp integer, x real, y real, qw real, qx real, qy real, qz real, vx real, vy real, acceleration_x real)"
    )
    connection.executemany(
        "insert into ego_pose values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                i * 100000,
                10.0,
                20.0 + i * 0.1,
                math.sqrt(0.5),
                0.0,
                0.0,
                math.sqrt(0.5),
                1.0,
                0.0,
                0.0,
            )
            for i in range(10)
        ],
    )
    connection.commit()
    connection.close()
    batch = read_ego_only(db, center_timestamp=300000, history_steps=3, future_steps=4)
    assert torch.allclose(
        batch["target"][0, :, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]), atol=1e-6
    )
    assert torch.allclose(batch["target"][0, :, 1], torch.zeros(4), atol=1e-6)


def test_current_ego_state_never_uses_future_sample(tmp_path):
    path = tmp_path / "causal-ego.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "create table ego_pose (timestamp integer, x real, y real, qw real, qx real, qy real, qz real, vx real, vy real, acceleration_x real)"
    )
    for i in range(10):
        if i != 3:
            connection.execute(
                "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?)",
                (i * 100000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            )
    connection.execute(
        "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?)",
        (250000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    )
    connection.execute(
        "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?)",
        (300001, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0),
    )
    connection.commit()
    connection.close()
    batch = read_ego_only(
        path, center_timestamp=300000, history_steps=3, future_steps=4
    )
    assert batch["ego_state"][0, 3] == 2.0
