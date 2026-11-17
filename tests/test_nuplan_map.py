import sqlite3
import struct
import pytest
import torch

pytest.importorskip("pyproj")
pytest.importorskip("shapely")
from pyproj import Transformer
from shapely.geometry import LineString
from pluto.nuplan_map import add_map_features
from pluto.synthetic import sample_batch


def test_map_probe_projects_lane_and_constructs_boundary_features(tmp_path):
    path = tmp_path / "map.gpkg"
    c = sqlite3.connect(path)
    c.execute(
        "create table baseline_paths (fid integer, geom blob, lane_fid integer, lane_connector_fid integer)"
    )
    c.execute(
        "create virtual table rtree_baseline_paths_geom using rtree(id,minx,maxx,miny,maxy)"
    )
    c.execute(
        "create table lanes_polygons (fid integer, left_boundary_fid integer, right_boundary_fid integer, speed_limit_mps real, lane_group_fid integer)"
    )
    c.execute(
        "create table lane_connectors (fid integer, speed_limit_mps real, lane_group_connector_fid integer, exit_lane_fid integer, entry_lane_fid integer)"
    )
    c.execute("create table boundaries (fid integer, geom blob)")
    east, north = 365845.0, 143242.0
    transformer = Transformer.from_crs(32648, 4326, always_xy=True)

    def geometry(y):
        points = [transformer.transform(east + x, north + y) for x in range(0, 21, 5)]
        return b"GP" + bytes([0, 1]) + struct.pack("<i", 4326) + LineString(points).wkb

    center, left, right = geometry(0.0), geometry(2.0), geometry(-2.0)
    lon, lat = transformer.transform(east, north)
    c.execute("insert into baseline_paths values (1,?,1,null)", (center,))
    c.execute(
        "insert into rtree_baseline_paths_geom values (1,?,?,?,?)",
        (lon - 0.001, lon + 0.001, lat - 0.001, lat + 0.001),
    )
    c.execute("insert into lanes_polygons values (1,1,2,10.,99)")
    c.executemany("insert into boundaries values (?,?)", [(1, left), (2, right)])
    c.commit()
    c.close()
    batch = sample_batch(1, 4)
    result = add_map_features(
        batch, path, (east, north, 0.0), epsg=32648, points=4, modes=3
    )
    assert result["map"].shape == (1, 1, 4, 10)
    assert result["reference"].shape == (1, 1, 4, 6)
    assert torch.allclose(result["map"][0, 0, :, 7], torch.full((4,), 2.0), atol=1e-3)
    assert torch.allclose(result["map"][0, 0, :, 9], torch.full((4,), -2.0), atol=1e-3)
    assert result["speed_known"].item() and result["speed_limit"].item() == 10.0
    assert result["traffic_light"].item() == 3
    routed = add_map_features(
        batch, path, (east, north, 0.0), epsg=32648, points=4, modes=3, route_ids=[99]
    )
    assert routed["on_route"].item() == 1 and routed["reference"].shape[1] == 1
    wrong_route = add_map_features(
        batch, path, (east, north, 0.0), epsg=32648, points=4, modes=3, route_ids=[123]
    )
    assert (
        wrong_route["on_route"].item() == 0 and wrong_route["reference"].shape[1] == 0
    )
    assert batch["map"].shape[-1] == 10 and batch["map"].shape[1] == 3
    changed = {key: value.clone() for key, value in batch.items()}
    changed["target"] += 100.0
    later = add_map_features(
        changed, path, (east, north, 0.0), epsg=32648, points=4, modes=3
    )
    assert torch.equal(result["reference"], later["reference"])
    assert torch.equal(result["map"], later["map"])
    c = sqlite3.connect(path)
    c.execute("insert into lane_connectors values (1, 8., 99, null, null)")
    c.execute("update baseline_paths set lane_fid=null, lane_connector_fid=1")
    c.commit()
    c.close()
    connector = add_map_features(
        batch,
        path,
        (east, north, 0.0),
        epsg=32648,
        points=4,
        modes=3,
        route_ids=[99],
        traffic_states={1: 2},
    )
    assert connector["on_route"].item() == 1 and connector["traffic_light"].item() == 2


def test_geopackage_decoder_rejects_bad_headers():
    from pluto.nuplan_map import _geometry

    with pytest.raises(ValueError, match="GeoPackage"):
        _geometry(b"not-a-geopackage")


def test_map_probe_requires_cpu_batch(tmp_path):
    # The adapter performs GeoPackage/pyproj work on CPU and does not silently
    # mix devices. A meta tensor models a non-CPU input without GPU hardware.
    batch = sample_batch(1, 4)
    batch["target"] = batch["target"].to("meta")
    with pytest.raises(ValueError, match="CPU"):
        add_map_features(batch, tmp_path / "unused.gpkg", (0.0, 0.0, 0.0), 32648)


def test_single_step_probe_reports_finite_gradients():
    from scripts.probe_nuplan_ego import diagnostic_step
    from pluto.model import Pluto, PlutoConfig

    config = PlutoConfig(
        dim=16,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        modes=3,
        future_steps=4,
        dropout=0.0,
    )
    model = Pluto(config)
    batch = sample_batch(1, 4)
    result = diagnostic_step(model, batch)
    assert result["steps"] == 1
    assert result["finite_loss"] and result["finite_gradients"]


def test_probe_cli_exposes_opt_in_optimization():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/probe_nuplan_ego.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--optimization-step" in result.stdout


def test_route_context_uses_latest_nonfuture_lidar_status(tmp_path):
    from pluto.nuplan_map import read_route_context

    path = tmp_path / "log.db"
    c = sqlite3.connect(path)
    c.execute("create table scene (token blob, roadblock_ids text)")
    c.execute("create table lidar_pc (token blob, timestamp integer, scene_token blob)")
    c.execute(
        "create table traffic_light_status (lidar_pc_token blob, lane_connector_id integer, status text)"
    )
    c.execute("insert into scene values (?,?)", (b"s", "99 100"))
    c.executemany(
        "insert into lidar_pc values (?,?,?)",
        [(b"p", 250000, b"s"), (b"f", 300001, b"s")],
    )
    c.executemany(
        "insert into traffic_light_status values (?,?,?)",
        [(b"p", 7, "green"), (b"f", 7, "red")],
    )
    c.commit()
    c.close()
    route, traffic = read_route_context(path, 300000)
    assert route == [99, 100]
    assert traffic == {7: 0}


def test_connected_reference_follows_lane_connector_and_route():
    from pluto.nuplan_map import LaneSegment, connected_reference_paths

    segments = [
        LaneSegment(("lane", 1), 10, LineString([(0.0, 0.0), (5.0, 0.0)])),
        LaneSegment(
            ("connector", 2),
            11,
            LineString([(5.0, 0.0), (6.0, 0.0)]),
            source_lane=1,
            target_lane=3,
        ),
        LaneSegment(("lane", 3), 12, LineString([(6.0, 0.0), (12.0, 0.0)])),
    ]
    paths = connected_reference_paths(
        segments, (1.0, 0.0, 0.0), route_ids=[10, 11, 12], max_references=1
    )
    assert len(paths) == 1
    assert paths[0].length == pytest.approx(11.0)
    assert paths[0].coords[0] == pytest.approx((1.0, 0.0))
    assert paths[0].coords[-1] == pytest.approx((12.0, 0.0))


def test_connected_reference_avoids_opposing_lanes_and_route_jumps():
    from pluto.nuplan_map import LaneSegment, connected_reference_paths

    segments = [
        LaneSegment(("lane", 1), 10, LineString([(5.0, 0.0), (0.0, 0.0)])),
        LaneSegment(("lane", 2), 30, LineString([(0.0, 1.0), (5.0, 1.0)])),
    ]
    assert not connected_reference_paths(
        segments, (1.0, 0.0, 0.0), route_ids=[10, 20], max_references=2
    )


def test_connected_paths_do_not_skip_a_route_roadblock():
    from pluto.nuplan_map import LaneSegment, connected_reference_paths

    segments = [
        LaneSegment(("lane", 1), 10, LineString([(0.0, 0.0), (5.0, 0.0)])),
        LaneSegment(
            ("connector", 2),
            30,
            LineString([(5.0, 0.0), (6.0, 0.0)]),
            source_lane=1,
            target_lane=3,
        ),
        LaneSegment(("lane", 3), 30, LineString([(6.0, 0.0), (12.0, 0.0)])),
    ]
    path = connected_reference_paths(
        segments, (1.0, 0.0, 0.0), route_ids=[10, 20, 30], max_references=1
    )[0]
    assert path.coords[-1] == pytest.approx((5.0, 0.0))
