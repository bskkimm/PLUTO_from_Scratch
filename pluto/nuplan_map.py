"""Read nearby nuPlan GeoPackage lanes into a heuristic local vector map."""

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import torch
from .losses import assign_modes


@dataclass(frozen=True)
class LaneSegment:
    """A route graph node expressed in projected world coordinates."""

    key: tuple
    roadblock: int
    line: object
    source_lane: int = None
    target_lane: int = None


def connected_reference_paths(
    segments, ego_pose, route_ids, max_references=6, max_length=120.0, max_gap=5.0
):
    """Stitch route-filtered lane/connector baselines in their directed order.

    This bounded heuristic uses connector exit→entry lane topology and clips
    the first segment at the ego projection. It is not the official scenario
    manager's route correction or reference-line algorithm.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import substring

    if max_references < 0 or max_length <= 0 or max_gap < 0:
        raise ValueError("invalid path limits")
    if not route_ids or max_references == 0:
        return []
    positions = {int(value): i for i, value in enumerate(route_ids)}
    nodes = {item.key: item for item in segments if item.roadblock in positions}
    adjacency = {key: [] for key in nodes}
    for key, item in nodes.items():
        if key[0] != "connector":
            continue
        source, target = ("lane", item.source_lane), ("lane", item.target_lane)
        if source in nodes:
            adjacency[source].append(key)
        if target in nodes:
            adjacency[key].append(target)
    for key in adjacency:
        adjacency[key].sort()
    origin = Point(ego_pose[0], ego_pose[1])
    starts = sorted(
        nodes.values(), key=lambda item: (item.line.distance(origin), item.key)
    )
    paths, signatures = [], set()

    def collect(key, coords, visited):
        if len(paths) >= max_references:
            return
        item = nodes[key]
        options = []
        for nxt in adjacency[key]:
            if nxt in visited:
                continue
            next_item = nodes[nxt]
            difference = positions[next_item.roadblock] - positions[item.roadblock]
            if difference not in (0, 1):
                continue
            next_coords = list(next_item.line.coords)
            if Point(coords[-1]).distance(Point(next_coords[0])) > max_gap:
                continue
            options.append((nxt, next_coords))
        if not options or LineString(coords).length >= max_length or len(visited) >= 32:
            line = LineString(coords)
            if line.length > max_length:
                line = substring(line, 0.0, max_length)
            if line.length > 0.01:
                signature = tuple(
                    (
                        round(line.interpolate(q, normalized=True).x, 1),
                        round(line.interpolate(q, normalized=True).y, 1),
                    )
                    for q in (0.0, 0.5, 1.0)
                )
                if signature not in signatures:
                    signatures.add(signature)
                    paths.append(line)
            return
        for nxt, next_coords in options:
            collect(nxt, coords + next_coords, visited | {nxt})
            if len(paths) >= max_references:
                break

    for item in starts:
        offset = item.line.project(origin)
        before = item.line.interpolate(max(0.0, offset - 0.25))
        after = item.line.interpolate(min(item.line.length, offset + 0.25))
        dot = (after.x - before.x) * math.cos(ego_pose[2]) + (
            after.y - before.y
        ) * math.sin(ego_pose[2])
        if dot <= 0:
            continue
        cropped = substring(item.line, offset, item.line.length)
        if cropped.geom_type != "LineString" or cropped.length < 0.01:
            continue
        collect(item.key, list(cropped.coords), {item.key})
        if len(paths) >= max_references:
            break
    return paths


def _geometry(blob):
    """Decode a GeoPackage binary geometry, including its optional envelope."""
    from shapely import from_wkb

    if not blob or blob[:2] != b"GP" or len(blob) < 8:
        raise ValueError("not a GeoPackage geometry")
    envelope = (blob[3] >> 1) & 7
    lengths = (0, 32, 48, 48, 64)
    if envelope >= len(lengths):
        raise ValueError("unsupported GeoPackage envelope")
    return from_wkb(blob[8 + lengths[envelope] :])


def add_map_features(
    batch,
    map_path,
    ego_pose,
    epsg,
    radius=120.0,
    points=20,
    max_polylines=128,
    max_references=6,
    modes=12,
    route_ids=None,
    traffic_states=None,
):
    """Return a batch with nearby lanes and optionally route-filtered references.

    Supports one scene [B=1]. `ego_pose` is world-frame (x,y,yaw) and EPSG
    describes the world coordinates. GeoPackage geometry is assumed WGS84.
    Boundaries missing from the map become zero offsets; speed/traffic fields
    are conservative unknowns when unavailable. Supplied route IDs filter
    reference candidates and use a bounded connectivity heuristic. This is
    not the official reference-path builder. No matching route gives no references.
    """
    if (
        radius <= 0
        or points < 2
        or max_polylines < 1
        or max_references < 0
        or modes < 2
    ):
        raise ValueError("invalid map sampling parameters")
    if batch["target"].shape[0] != 1:
        raise ValueError("map probe currently supports one scene")
    if any(value.device.type != "cpu" for value in batch.values()):
        raise ValueError("map preprocessing requires CPU tensors")
    try:
        from pyproj import Transformer
        from shapely.geometry import LineString, Point
    except ImportError as exc:
        raise RuntimeError("map probing requires pyproj and shapely") from exc
    east, north, angle = ego_pose
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    to_world = Transformer.from_crs(4326, epsg, always_xy=True)
    minlon, minlat, maxlon, maxlat = to_ll.transform_bounds(
        east - radius, north - radius, east + radius, north + radius, densify_pts=21
    )
    path = Path(map_path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT b.fid,b.geom,b.lane_fid,b.lane_connector_fid,
                   left_b.geom,right_b.geom,COALESCE(l.speed_limit_mps,c.speed_limit_mps),
                   COALESCE(l.lane_group_fid,c.lane_group_connector_fid),
                   c.exit_lane_fid,c.entry_lane_fid
            FROM baseline_paths b JOIN rtree_baseline_paths_geom r ON b.fid=r.id
            LEFT JOIN lanes_polygons l ON b.lane_fid=l.fid
            LEFT JOIN lane_connectors c ON b.lane_connector_fid=c.fid
            LEFT JOIN boundaries left_b ON l.left_boundary_fid=left_b.fid
            LEFT JOIN boundaries right_b ON l.right_boundary_fid=right_b.fid
            WHERE r.maxx>=? AND r.minx<=? AND r.maxy>=? AND r.miny<=?
            """,
            (minlon, maxlon, minlat, maxlat),
        ).fetchall()

    def project(blob):
        geometry = _geometry(blob)
        if geometry.geom_type != "LineString" or geometry.is_empty:
            return None
        coords = list(geometry.coords)
        x, y = to_world.transform([p[0] for p in coords], [p[1] for p in coords])
        line = LineString(zip(x, y))
        return line if line.length > 0.01 else None

    nearby = []
    for (
        fid,
        blob,
        lane_id,
        connector_id,
        left,
        right,
        speed,
        roadblock_id,
        source_lane,
        target_lane,
    ) in rows:
        center = project(blob)
        if center is None or center.distance(Point(east, north)) > radius:
            continue
        left_line, right_line = (
            project(left) if left else None,
            project(right) if right else None,
        )
        nearby.append(
            (
                center.distance(Point(east, north)),
                fid,
                center,
                lane_id,
                connector_id,
                left_line,
                right_line,
                speed,
                roadblock_id,
                source_lane,
                target_lane,
            )
        )
    nearby.sort(key=lambda item: (item[0], item[1]))
    nearby = nearby[:max_polylines]
    cos0, sin0 = (
        torch.cos(torch.tensor(angle)).item(),
        torch.sin(torch.tensor(angle)).item(),
    )

    def local(point):
        dx, dy = point.x - east, point.y - north
        return torch.tensor(
            [cos0 * dx + sin0 * dy, -sin0 * dx + cos0 * dy], dtype=torch.float32
        )

    def sample(line):
        return torch.stack(
            [
                local(line.interpolate(j / points, normalized=True))
                for j in range(points + 1)
            ]
        )

    (
        maps,
        poses,
        refs,
        ref_poses,
        raw_refs,
        map_types,
        speeds,
        known,
        on_route,
        lights,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    route_set = None if route_ids is None else set(route_ids)
    traffic_states = traffic_states or {}
    segments = []
    for (
        _,
        _,
        line,
        lane_id,
        connector_id,
        left,
        right,
        speed,
        roadblock_id,
        source_lane,
        target_lane,
    ) in nearby:
        key = ("lane", lane_id) if lane_id is not None else ("connector", connector_id)
        segments.append(LaneSegment(key, roadblock_id, line, source_lane, target_lane))
        center = sample(line)
        vector = center[1:] - center[:-1]
        yaw = torch.atan2(vector[:, 1], vector[:, 0])
        origin = center[points // 2]

        def boundary_offset(boundary):
            if boundary is None:
                return torch.zeros(points, 2)
            sampled = sample(boundary)
            if torch.linalg.vector_norm(
                sampled[0] - center[0]
            ) > torch.linalg.vector_norm(sampled[-1] - center[0]):
                sampled = sampled.flip(0)
            return sampled[:-1] - center[:-1]

        maps.append(
            torch.cat(
                (
                    center[:-1] - origin,
                    vector,
                    torch.stack((yaw.cos(), yaw.sin()), -1),
                    boundary_offset(left),
                    boundary_offset(right),
                ),
                -1,
            )
        )
        poses.append(torch.cat((origin, yaw[points // 2 : points // 2 + 1])))
        refs.append(
            torch.cat(
                (
                    center[:-1] - center[0],
                    vector,
                    torch.stack((yaw.cos(), yaw.sin()), -1),
                ),
                -1,
            )
        )
        ref_poses.append(torch.cat((center[0], yaw[:1])))
        raw_refs.append(center[:-1])
        map_types.append(0 if lane_id is not None else 1)
        known.append(speed is not None)
        speeds.append(float(speed) if speed is not None else 0.0)
        on_route.append(int(route_set is not None and roadblock_id in route_set))
        status = (
            int(traffic_states.get(connector_id, 3)) if connector_id is not None else 3
        )
        if status not in (0, 1, 2, 3):
            raise ValueError("traffic state must be 0–3")
        lights.append(status)
    n = len(maps)
    if route_set is not None:
        reference_lines = connected_reference_paths(
            segments,
            ego_pose,
            route_ids,
            max_references=max_references,
            max_length=radius,
        )
        refs, ref_poses, raw_refs = [], [], []
        for line in reference_lines:
            center = sample(line)
            vector = center[1:] - center[:-1]
            yaw = torch.atan2(vector[:, 1], vector[:, 0])
            refs.append(
                torch.cat(
                    (
                        center[:-1] - center[0],
                        vector,
                        torch.stack((yaw.cos(), yaw.sin()), -1),
                    ),
                    -1,
                )
            )
            ref_poses.append(torch.cat((center[0], yaw[:1])))
            raw_refs.append(center[:-1])
        reference_indices = list(range(len(refs)))
    else:
        reference_indices = list(range(min(len(refs), max_references)))
    r = len(reference_indices)
    out = {key: value.clone() for key, value in batch.items()}
    out["map"] = torch.stack(maps)[None] if n else torch.zeros(1, 0, points, 10)
    out["map_mask"] = torch.ones(1, n, points, dtype=torch.bool)
    out["map_pose"] = torch.stack(poses)[None] if n else torch.zeros(1, 0, 3)
    out["map_type"] = torch.tensor([map_types], dtype=torch.long)
    out["on_route"] = torch.tensor([on_route], dtype=torch.long)
    out["traffic_light"] = torch.tensor([lights], dtype=torch.long)
    out["speed_limit"] = torch.tensor([speeds], dtype=torch.float32)
    out["speed_known"] = torch.tensor([known], dtype=torch.bool)
    out["reference"] = (
        torch.stack([refs[i] for i in reference_indices])[None]
        if r
        else torch.zeros(1, 0, points, 6)
    )
    out["reference_pose"] = (
        torch.stack([ref_poses[i] for i in reference_indices])[None]
        if r
        else torch.zeros(1, 0, 3)
    )
    out["reference_mask"] = torch.ones(1, r, points, dtype=torch.bool)
    if r and out["target_mask"].any():
        last = out["target_mask"][0].nonzero()[-1, 0]
        out["mode_label"] = assign_modes(
            out["target"][:, last, :2],
            torch.stack([raw_refs[i] for i in reference_indices])[None],
            out["reference_mask"],
            modes,
        )
    else:
        out["mode_label"] = torch.zeros(1, dtype=torch.long)
    return out


def read_route_context(log_path, timestamp, tolerance_us=100000):
    """Read scene route IDs and traffic states from a nonfuture lidar frame.

    `None` route means the log has no route information; an empty list is a
    supplied empty route. Traffic mapping follows nuPlan's enum: green=0,
    yellow=1, red=2, unknown=3. This does not reconstruct route topology.
    """
    if tolerance_us < 0:
        raise ValueError("tolerance must be nonnegative")
    path = Path(log_path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT pc.token,s.roadblock_ids FROM lidar_pc pc LEFT JOIN scene s ON pc.scene_token=s.token "
            "WHERE pc.timestamp BETWEEN ? AND ? ORDER BY pc.timestamp DESC LIMIT 1",
            (timestamp - tolerance_us, timestamp),
        ).fetchone()
        if row is None:
            raise ValueError("no nonfuture lidar frame near requested timestamp")
        route = [int(value) for value in row[1].split()] if row[1] is not None else None
        lights = connection.execute(
            "SELECT lane_connector_id,status FROM traffic_light_status WHERE lidar_pc_token=?",
            (row[0],),
        ).fetchall()
    code = {"green": 0, "yellow": 1, "red": 2, "unknown": 3}
    return route, {
        int(connector): code.get(str(status).lower(), 3) for connector, status in lights
    }
