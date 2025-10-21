import pytest
from mom6_bathy.aux import get_avg_resolution, get_avg_resolution_km, longitude_slicer
from utils import on_cisl_machine
import xarray as xr
import numpy as np 

def test_avg_resolution():
    """Test the average resolution calculation for a grid."""

    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")

    t232_avg_res = get_avg_resolution("/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc")
    assert 0.49 < t232_avg_res < 0.50, "Average resolution for tx2_3v2 should be around 0.5 degrees"

    t232_avg_res_km = get_avg_resolution_km("/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc")
    assert 40.0 < t232_avg_res_km < 41.0, "Average resolution for tx2_3v2 should be around 40 km"

def test_longitude_slicer():
    with pytest.raises(AssertionError):
        nx, ny, nt = 4, 14, 5

        latitude_extent = (10, 20)
        longitude_extent = (12, 18)

        dims = ["silly_lat", "silly_lon", "time"]

        dλ = (longitude_extent[1] - longitude_extent[0]) / 2

        data = xr.DataArray(
            np.random.random((ny, nx, nt)),
            dims=dims,
            coords={
                "silly_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
                "silly_lon": np.array(
                    [
                        longitude_extent[0],
                        longitude_extent[0] + 1.5 * dλ,
                        longitude_extent[0] + 2.6 * dλ,
                        longitude_extent[1],
                    ]
                ),
                "time": np.linspace(0, 1000, nt),
            },
        )

        longitude_slicer(data, longitude_extent, "silly_lon")
