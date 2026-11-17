import sqlite3
import torch
from pluto.nuplan_actors import read_ego_and_actors


def test_actor_selection_uses_present_frame_not_future(tmp_path):
    path = tmp_path / "mini.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "create table ego_pose (timestamp integer, x real, y real, qw real, qx real, qy real, qz real, vx real, vy real, acceleration_x real)"
    )
    connection.execute("create table lidar_pc (token blob, timestamp integer)")
    connection.execute(
        "create table lidar_box (lidar_pc_token blob, track_token blob, x real, y real, yaw real, vx real, vy real, width real, length real)"
    )
    connection.execute("create table track (token blob, category_token blob)")
    connection.execute("create table category (token blob, name text)")
    connection.execute("insert into category values (?,?)", (b"v", "vehicle"))
    connection.executemany(
        "insert into track values (?,?)", [(b"a", b"v"), (b"b", b"v")]
    )
    for i in range(10):
        ts = i * 100000
        token = bytes([i])
        connection.execute(
            "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?)",
            (ts, i * 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        )
        connection.execute("insert into lidar_pc values (?,?)", (token, ts))
        connection.execute(
            "insert into lidar_box values (?,?,?,?,?,?,?,?,?)",
            (token, b"a", 5.0 + i * 0.1, 0.0, 0.0, 1.0, 0.0, 2.0, 4.0),
        )
        if i > 3:
            connection.execute(
                "insert into lidar_box values (?,?,?,?,?,?,?,?,?)",
                (token, b"b", 8.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0),
            )
    connection.commit()
    connection.close()
    result = read_ego_and_actors(
        path, center_timestamp=300000, history_steps=3, future_steps=4
    )
    assert result["agents"].shape == (1, 2, 2, 9)
    assert result["agent_type"].tolist() == [[0, 1]]
    assert torch.allclose(result["agent_pose"][0, 1, :2], torch.tensor([5.0, 0.0]))
    assert torch.allclose(
        result["agent_target"][0, 0, :, 0],
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        atol=1e-6,
    )
    assert result["agent_target_mask"].all()
    assert result["reference"].shape[1] == 0


def test_current_actor_frame_never_comes_from_future(tmp_path):
    path = tmp_path / "causality.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "create table ego_pose (timestamp integer, x real, y real, qw real, qx real, qy real, qz real, vx real, vy real, acceleration_x real)"
    )
    connection.execute("create table lidar_pc (token blob, timestamp integer)")
    connection.execute(
        "create table lidar_box (lidar_pc_token blob, track_token blob, x real, y real, yaw real, vx real, vy real, width real, length real)"
    )
    connection.execute("create table track (token blob, category_token blob)")
    connection.execute("create table category (token blob, name text)")
    connection.execute("insert into category values (?,?)", (b"v", "vehicle"))
    connection.execute("insert into track values (?,?)", (b"a", b"v"))
    for i in range(10):
        ts = i * 100000
        connection.execute(
            "insert into ego_pose values (?,?,?,?,?,?,?,?,?,?)",
            (ts, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        if i != 3:
            connection.execute("insert into lidar_pc values (?,?)", (bytes([i]), ts))
    connection.execute("insert into lidar_pc values (?,?)", (b"p", 250000))
    connection.execute("insert into lidar_pc values (?,?)", (b"f", 300001))
    connection.execute(
        "insert into lidar_box values (?,?,?,?,?,?,?,?,?)",
        (b"f", b"a", 5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0),
    )
    connection.commit()
    connection.close()
    batch = read_ego_and_actors(
        path, center_timestamp=300000, history_steps=3, future_steps=4
    )
    assert batch["agents"].shape[1] == 1
