import numpy as np
import pytest
from mom6_bathy.aux import gc_dist, gc_tarea, gc_qarea

def test_gc_dist():
    # Test case with scalar inputs
    lat1 = 0.0
    lon1 = 0.0
    lat2 = 0.0
    lon2 = 45.0
    expected_result = np.pi / 4.0
    result = gc_dist(lat1, lon1, lat2, lon2)
    assert result == pytest.approx(expected_result)

    # Test case with array inputs
    lat1 = np.array([0.0, 0.0])
    lon1 = np.array([0.0, 0.0])
    lat2 = np.array([45.0, 45.0])
    lon2 = np.array([0.0, 0.0])
    expected_result = np.array([np.pi / 4.0, np.pi / 4.0])
    result = gc_dist(lat1, lon1, lat2, lon2)
    assert result == pytest.approx(expected_result)

def test_gc_tarea():
    # Test case with scalar inputs
    lat1 = 0.0
    lon1 = 0.0
    lat2 = 45.0
    lon2 = 45.0
    lat3 = 90.0
    lon3 = 0.0
    result = gc_tarea(lat1, lon1, lat2, lon2, lat3, lon3)
    assert result == pytest.approx(0.2526802551420787)

    # Test case with array inputs
    lat1 = np.array([0.0, 0.0])
    lon1 = np.array([0.0, 0.0])
    lat2 = np.array([45.0, 45.0])
    lon2 = np.array([45.0, 45.0])
    lat3 = np.array([90.0, 90.0])
    lon3 = np.array([0.0, 0.0])
    expected_result = np.array([0.2526802551420787, 0.2526802551420787])
    result = gc_tarea(lat1, lon1, lat2, lon2, lat3, lon3)
    assert result == pytest.approx(expected_result)

def test_gc_qarea():
    # Test case with 1D inputs
    lat = np.array([-0.5, -0.5, 0.5, 0.5])
    lon = np.array([-0.5, 0.5, 0.5, -0.5])
    expected_result = (2 * np.pi / 360.0)**2
    result = gc_qarea(lat, lon)
    assert np.isclose(result, expected_result, rtol=1e-3)


if __name__ == "__main__":
    test_gc_dist()
    test_gc_tarea()
    test_gc_qarea()
