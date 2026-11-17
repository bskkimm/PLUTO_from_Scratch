import pytest
import torch

pytest.importorskip("shapely")
from shapely.geometry import Polygon
from pluto.raster import rasterize_scene


def test_raster_coordinates_occupancy_and_overlap():
    drivable = [Polygon([(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)])]
    obstacles = [Polygon([(0.4, 0.4), (1.6, 0.4), (1.6, 1.6), (0.4, 1.6)])]
    road, occupied = rasterize_scene(
        drivable, obstacles, height=5, width=5, resolution=1.0
    )
    assert road.dtype == occupied.dtype == torch.bool
    assert road[0, 2, 2]
    assert occupied[0, 1, 3]
    assert not occupied[0, 3, 3]
    assert not road[0, 0, 0]


def test_raster_rejects_bad_size_and_resolution():
    with pytest.raises(ValueError):
        rasterize_scene([], [], height=1, width=5, resolution=1.0)
    with pytest.raises(ValueError):
        rasterize_scene([], [], height=5, width=5, resolution=0.0)
