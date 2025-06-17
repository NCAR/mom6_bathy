from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from mom6_bathy.chl import interpolate_and_fill_seawifs
from .utils import on_cisl_machine
import pytest
import os


def test_chl():
    """Test the creation of chl files."""
    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")
    # attempt to create a regional grid object from scratch
    grid = Grid(
        resolution=0.01,
        xstart=278.0,
        lenx=1.0,
        ystart=7.0,
        leny=1.0,
        name="panama1",
    )

    grid.name = "rand"
    # create a corresponding bathymetry object
    topo = Topo(grid, min_depth=10.0)

    # set the bathymetry to a flat bottom
    topo.set_flat(D=2000.0)

    interpolate_and_fill_seawifs(
        grid,
        topo,
        processed_seawifs_path="/glade/campaign/cesm/cesmdata/cseg/inputdata/ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc",
    )

    assert os.path.exists(
        "/glade/campaign/cesm/cesmdata/cseg/inputdata/ocn/mom/croc/chl/data/seawifs-clim-1997-2010-rand.nc"
    )
