import os
import numpy as np
import xarray as xr
import pytest
from .grid import Grid

# CrocoDash/visualCaseGen/external/mom6_bathy/mom6_bathy/test_grid.py


@pytest.fixture
def simple_grid():
    # Create a simple 2x2 grid for testing
    grid = Grid(
        lenx=2.0,
        leny=2.0,
        nx=2,
        ny=2,
        resolution=1.0,
        xstart=0.0,
        ystart=0.0,
        name="testgrid",
        save_on_create=False
    )
    return grid

def test_grid_properties(simple_grid):
    grid = simple_grid
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid.lenx == 2.0
    assert grid.leny == 2.0
    assert grid.resolution == 1.0
    assert grid.name == "testgrid"

def test_grid_sanitize_name():
    g = Grid(
        lenx=2.0, leny=2.0, nx=2, ny=2, resolution=1.0, name="bad name!@#", save_on_create=False
    )
    assert g.name == "bad_name_"

def test_grid_get_indices(simple_grid):
    grid = simple_grid
    # Should return a valid index for the center
    j, i = grid.get_indices(grid.tlat.values[0, 0], grid.tlon.values[0, 0])
    assert 0 <= j < grid.ny
    assert 0 <= i < grid.nx

def test_grid_is_rectangular(simple_grid):
    assert simple_grid.is_rectangular()

def test_grid_slice(simple_grid):
    sub = simple_grid[0:1, 0:1]
    assert isinstance(sub, Grid)
    assert sub.nx == 1
    assert sub.ny == 1

def test_grid_supergrid_setter(simple_grid):
    sg = simple_grid.supergrid
    simple_grid.supergrid = sg  # Should not raise

def test_grid_to_netcdf_and_from_netcdf(tmp_path, simple_grid):
    path = tmp_path / "testgrid.nc"
    simple_grid.to_netcdf(str(path))
    assert os.path.exists(path)
    loaded = Grid.from_netcdf(str(path))
    assert loaded.nx == simple_grid.nx
    assert loaded.ny == simple_grid.ny
    assert loaded.name == simple_grid.name

def test_grid_save_and_load_metadata(tmp_path, simple_grid):
    json_path = tmp_path / "meta.json"
    simple_grid.save_metadata(str(json_path), message="test", ncfile="file.nc")
    assert os.path.exists(json_path)
    meta = Grid.load_metadata(str(json_path))
    assert meta["message"] == "test"
    assert meta["ncfile"] == "file.nc"

def test_grid_list_metadata_files(tmp_path, simple_grid):
    # Create a fake metadata file
    grid_dir = tmp_path / "Grids" / "testgrid_2x2"
    os.makedirs(grid_dir, exist_ok=True)
    json_path = grid_dir / "grid_testgrid.json"
    with open(json_path, "w") as f:
        f.write("{}")
    files = Grid.list_metadata_files(tmp_path / "Grids")
    assert any("grid_testgrid.json" in f for f in files)