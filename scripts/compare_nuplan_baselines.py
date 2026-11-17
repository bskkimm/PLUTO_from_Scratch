"""Compare local GeoPackage geometry decoding with the nuPlan devkit.

Requires the nuPlan devkit and its map dependencies to be installed separately.
Reads a single existing map and log without initializing the devkit's downloader.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from pyproj import Transformer
from shapely.geometry import LineString, Point
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

from pluto.nuplan_map import _geometry


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(map_root, location, version, log, timestamp, radius):
    root = Path(map_root).resolve(strict=True)
    map_path = (root / location / version / "map.gpkg").resolve(strict=True)
    log = Path(log).resolve(strict=True)

    class ReadOnlyMaps(GPKGMapsDB):
        # The normal constructor creates lock files and may download maps.
        def __init__(self):
            self._map_root = str(root)

        def get_version(self, requested_location):
            if requested_location != location:
                raise ValueError("unexpected map location")
            return version

        def _wait_for_expected_filesize(self, path_on_disk, requested_location):
            pass

    with sqlite3.connect(log.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT timestamp,x,y FROM ego_pose ORDER BY ABS(timestamp - ?) LIMIT 1",
            (timestamp,),
        ).fetchone()
    if row is None:
        raise ValueError("log has no ego poses")
    selected_timestamp, x, y = row
    layer = ReadOnlyMaps().load_vector_layer(location, "baseline_paths")
    transform = Transformer.from_crs(4326, layer.crs, always_xy=True)
    with sqlite3.connect(map_path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT fid,geom FROM baseline_paths").fetchall()
    errors = []
    origin = Point(x, y)
    for fid, blob in rows:
        geometry = _geometry(blob)
        if geometry.geom_type != "LineString" or geometry.is_empty:
            continue
        coords = list(geometry.coords)
        east, north = transform.transform(
            [point[0] for point in coords], [point[1] for point in coords]
        )
        local = LineString(zip(east, north))
        if local.distance(origin) <= radius:
            errors.append(local.hausdorff_distance(layer.loc[str(fid)].geometry))
    return {
        "map_sha256": sha256(map_path),
        "log_sha256": sha256(log),
        "location": location,
        "map_version": version,
        "requested_timestamp_us": timestamp,
        "selected_timestamp_us": selected_timestamp,
        "radius_m": radius,
        "projected_crs": str(layer.crs),
        "baseline_count": len(errors),
        "max_hausdorff_distance_m": max(errors) if errors else None,
        "limitation": "Geometry projection only; this does not validate route generation or planning.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--radius", type=float, default=120.0)
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("radius must be positive")
    print(json.dumps(compare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
