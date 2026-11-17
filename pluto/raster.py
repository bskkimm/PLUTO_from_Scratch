"""CPU rasterization of caller-provided local-frame polygons."""

import torch


def rasterize_scene(
    drivable_polygons, obstacle_polygons, height=500, width=500, resolution=0.2
):
    """Rasterize polygon interiors at cell centers into bool [1,H,W] masks.

    Polygon coordinates use ego-frame metres: x increases with columns and y
    decreases with rows. Image origin is centered at ((W-1)/2,(H-1)/2).
    Geometry is non-differentiable preprocessing. Invalid polygons are rejected
    rather than repaired; callers are responsible for map and obstacle semantics.
    """
    if height < 2 or width < 2 or height > 2000 or width > 2000 or resolution <= 0:
        raise ValueError("invalid raster dimensions or resolution")
    try:
        import numpy as np
        from shapely import contains_xy
    except ImportError as exc:
        raise RuntimeError("rasterization requires numpy and shapely >=2") from exc
    xs = (np.arange(width, dtype=np.float64) - (width - 1) / 2) * resolution
    ys = -((np.arange(height, dtype=np.float64) - (height - 1) / 2) * resolution)
    xx, yy = np.meshgrid(xs, ys)
    result = []
    for geometries in (drivable_polygons, obstacle_polygons):
        mask = np.zeros((height, width), dtype=bool)
        for geometry in geometries:
            if geometry.is_empty:
                continue
            if not geometry.is_valid or geometry.geom_type not in (
                "Polygon",
                "MultiPolygon",
            ):
                raise ValueError("expected valid polygon geometry")
            minx, miny, maxx, maxy = geometry.bounds
            region = (xx >= minx) & (xx <= maxx) & (yy >= miny) & (yy <= maxy)
            mask[region] |= contains_xy(geometry, xx[region], yy[region])
        result.append(torch.from_numpy(mask).unsqueeze(0))
    return tuple(result)
