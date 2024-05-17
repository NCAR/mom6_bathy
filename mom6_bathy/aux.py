"""Auxiliary functions for grid property computations."""

import numpy as np


def gc_dist(lat1, lon1, lat2, lon2, radius=1.0):
    """Great circle distance between two points on a sphere.

    Parameters
    ----------
    lat1 : float or array_like
        Latitude(s) of the first point(s). (degrees)
    lon1 : float or array_like
        Longitude(s) of the first point(s). (degrees)
    lat2 : float or array_like
        Latitude(s) of the second point(s). (degrees)
    lon2 : float or array_like
        Longitude(s) of the second point(s). (degrees)
    radius : float, optional
        Radius of the sphere. Default is 1.0.

    Returns
    -------
    float or array_like
        Great circle distance(s) between the points. (radians)
    """

    return radius * np.arccos(
        np.sin(np.radians(lat1)) * np.sin(np.radians(lat2))
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.cos(np.radians(lon1 - lon2))
    )


def gc_tarea(lat1, lon1, lat2, lon2, lat3, lon3, radius=1.0):
    """Computes the area of a triangular patch whose vertices are given by
    degrees of latitude and longitude on a sphere.

    Parameters
    ----------
    lat1 : float or array_like
        Latitude(s) of the first vertex/vertices. (degrees)
    lon1 : float
        Longitude(s) of the first vertex/vertices. (degrees)
    lat2 : float
        Latitude(s) of the second vertex/vertices. (degrees)
    lon2 : float
        Longitude(s) of the second vertex/vertices. (degrees)
    lat3 : float
        Latitude(s) of the third vertex/vertices. (degrees)
    lon3 : float
        Longitude(s) of the third vertex/vertices. (degrees)
    radius : float, optional
        Radius of the sphere. Default is 1.0.

    Returns
    -------
    float or array_like
        Area(s) of the triangular patch(es). (radians)
    """

    # Sanity checks
    if np.isscalar(lat1):
        assert (
            np.isscalar(lon1)
            and np.isscalar(lat2)
            and np.isscalar(lon2)
            and np.isscalar(lat3)
            and np.isscalar(lon3)
        ), "lat and lon must be all scalars or all arrays"
    else:
        assert (
            lat1.shape == lat2.shape and lat2.shape == lat3.shape
        ), "lat and lon must have the same shape"
        assert len(lat1.shape) == 1, "lat and lon must be 1d arrays"

    a = gc_dist(lat1, lon1, lat2, lon2, radius)
    b = gc_dist(lat2, lon2, lat3, lon3, radius)
    c = gc_dist(lat3, lon3, lat1, lon1, radius)
    s = 0.5 * (a + b + c)
    return radius**2 * np.arcsin(
        np.sqrt(np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c))
    )


def gc_qarea(lat, lon, radius=1.0):
    """Computes the area of a quadrilateral patch whose vertices are given by
    degrees of latitude and longitude on a sphere.

    Parameters
    ----------
    lat : array_like
        Array of latitudes of the vertices. (degrees)
    lon : array_like
        Array of longitudes of the vertices. (degrees)
    radius : float, optional
        Radius of the sphere. Default is 1.0.
    """

    # Check if 2d:
    if len(lat.shape) == 2:
        assert (
            lat.shape[1] == 4 and lon.shape[1] == 4
        ), "lat and lon must have shape (n, 4)"

        # Compute the area of the two triangles
        area1 = gc_tarea(
            lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1], lat[:, 2], lon[:, 2], radius
        )
        area2 = gc_tarea(
            lat[:, 0], lon[:, 0], lat[:, 2], lon[:, 2], lat[:, 3], lon[:, 3], radius
        )
        return area1 + area2

    elif len(lat.shape) == 1:
        assert len(lat) == 4 and len(lon) == 4, "lat and lon must have length 4"

        # Compute the area of the two triangles
        area1 = gc_tarea(lat[0], lon[0], lat[1], lon[1], lat[2], lon[2], radius)
        area2 = gc_tarea(lat[0], lon[0], lat[2], lon[2], lat[3], lon[3], radius)
        return area1 + area2

    else:
        raise ValueError("lat and lon must be 1d or 2d arrays of length 4")
