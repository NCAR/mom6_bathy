import pytest
from mom6_bathy.aux import get_avg_resolution, get_avg_resolution_km
from utils import on_cisl_machine

def test_avg_resolution():
    """Test the average resolution calculation for a grid."""

    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")

    t232_avg_res = get_avg_resolution("/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc")
    assert 0.49 < t232_avg_res < 0.50, "Average resolution for tx2_3v2 should be around 0.5 degrees"

    t232_avg_res_km = get_avg_resolution_km("/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc")
    assert 40.0 < t232_avg_res_km < 41.0, "Average resolution for tx2_3v2 should be around 40 km"