import pytest
from mom6_bathy._supergrid import *
import numpy as np

@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0,10], [0,10]),
    ],
)
def test_even_spacing_hgrid(lat, lon):
    assert isinstance(EvenSpacingSupergrid(lon[0],lon[1]-lon[0], lat[0], lat[1]-lat[0],0.05), EvenSpacingSupergrid)

@pytest.mark.parametrize(
    ("lat", "lon", "true_area"),
    [
        (np.meshgrid(np.linspace(0, 90, 5), np.linspace(0, 90, 5)), 1 / 8 * (4 * np.pi)),# create a lat-lon mesh that covers 1/4 of the North Hemisphere

        (np.meshgrid(np.linspace(-45, 45, 5), np.linspace(-90, 90, 5)), 1 / 4 * (4 * np.pi)),# create a lat-lon mesh that covers 1/4 of the whole globe
    ],
)   
def test_quadrilateral_areas(lat, lon, true_area):
    assert np.isclose(np.sum(quadrilateral_areas(lat, lon)), true_area)
    
@pytest.mark.parametrize(
    ("lat", "lon", "true_xyz"),
    [
        (0, 0, (1, 0, 0)),
        (90, 0, (0, 0, 1)),
        (0, 90, (0, 1, 0)),
        (-90, 0, (0, 0, -1)),
    ],
)
def test_latlon_to_cartesian(lat, lon, true_xyz):
    assert np.isclose(latlon_to_cartesian(lat, lon), true_xyz).all()


def test_quadrilateral_area_exception():
    v1 = np.dstack(latlon_to_cartesian(0, 0, R=2))
    v2 = np.dstack(latlon_to_cartesian(90, 0, R=2))
    v3 = np.dstack(latlon_to_cartesian(0, 90, R=2))
    v4 = np.dstack(latlon_to_cartesian(-90, 0, R=2.1))
    with pytest.raises(ValueError) as excinfo:
        quadrilateral_area(v1, v2, v3, v4)

    assert str(excinfo.value) == "vectors provided must have the same length"

@pytest.mark.parametrize(
    ("v1", "v2", "v3", "v4", "true_area"),
    [
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, -90)),
            np.pi,
        ),
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(-90, 0)),
            np.pi,
        ),
    ],
)
def test_quadrilateral_area(v1, v2, v3, v4, true_area):
    assert np.isclose(quadrilateral_area(v1, v2, v3, v4), true_area)