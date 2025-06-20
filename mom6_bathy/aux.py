"""Auxiliary functions for grid property computations."""

from pathlib import Path
import xarray as xr
import numpy as np


def get_mesh_dimensions(mesh):
    """Given an ESMF mesh where the grid metrics are stored in 1D (flattened) arrays,
    compute the dimensions of the 2D grid and return them as nx, ny.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.
    
    Returns
    -------
    nx : int
        Number of points in the x-direction.
    ny : int
        Number of points in the y-direction.
    """

    if not isinstance(mesh, xr.Dataset):
        assert isinstance(mesh, (Path, str)) and Path(mesh).exists(), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    centerCoords = mesh['centerCoords'].values

    x0, y0 = centerCoords[0]  # First coordinate
    x0 = (x0 + 360) % 360  # Normalize longitude

    coords = centerCoords[:, :2]
    x, y = coords[:, 0], coords[:, 1]

    # Compute distances in bulk
    dists = (np.mod(x + 360 - x0, 360)) ** 2 + (y - y0) ** 2

    # Diff of distances
    diff = np.diff(dists)
    
    # Index of the first decrease in distances:
    i = np.where(diff < 0)[0][0]
    # Index of the first increase after the first decrease:
    i = np.where(diff[i:] > 0)[0][0] + i 

    nx = i
    ny = len(centerCoords) // nx

    # Check that nx is indeed nx and not ny, and if not, swap them
    if np.abs(np.mod(coords[nx//2, 0] - x0, 360)) < np.abs(coords[nx//2, 1] - y0):
        nx, ny = ny, nx

    assert nx * ny == len(centerCoords), "nx*ny must match the number of points in the mesh"
    return nx, ny



def _lonlat_to_unitvec(lon, lat):
    """Convert longitude and latitude to unit vectors.
    This functions is used for area computation

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude in degrees.
    lat : float or np.ndarray
        Latitude in degrees.

    Returns
    -------
    result : np.ndarray
        Unit vectors (x, y, z)
    """
    



def _spherical_angle(a, b):
    """Compute angle between unit vectors a and b on a sphere.
    This function is used for area computation.

    Parameters
    ----------
    a : np.ndarray, shape (..., 3)
        First unit vector (x, y, z).
    b : np.ndarray, shape (..., 3)
        Second unit vector (x, y, z).

    Returns
    -------
    angle : np.ndarray
        Angle in radians.
    """

    cross = np.cross(a, b)
    sin_angle = np.linalg.norm(cross, axis=-1)
    cos_angle = np.sum(a * b, axis=-1)
    return np.arctan2(sin_angle, cos_angle)

def _tri_area(u, v, w):
    """Vectorized spherical triangle area using L'Huilier's theorem.

    Parameters:
    -----------
    u, v, w: np.ndarray of shape (..., 3)
        Arrays of unit vectors representing triangle vertices.

    Returns:
    --------
    np.ndarray of shape (...,): Areas of the spherical triangles.
    """
    a = _spherical_angle(v, w)
    b = _spherical_angle(u, w)
    c = _spherical_angle(u, v)
    s = 0.5 * (a + b + c)

    t = np.tan(0.5 * s) * np.tan(0.5 * (s - a)) * \
        np.tan(0.5 * (s - b)) * np.tan(0.5 * (s - c))

    area = np.abs(4.0 * np.arctan(np.sqrt(np.abs(t))))
    return area

def _great_circle_area(polygon_unitvecs):
    """
    Compute area of multiple spherical polygons using triangulation from vertex 0.

    Parameters:
    -----------
    polygon_unitvecs: np.ndarray of shape (n_poly, n_verts, 3)

    Returns:
    --------
    np.ndarray of shape (n_poly,): Areas of the spherical polygons
    """
    n_poly, n_verts, _ = polygon_unitvecs.shape
    if n_verts < 3:
        return np.zeros(n_poly)

    pnt0 = polygon_unitvecs[:, 0:1, :]         # shape (n_poly, 1, 3)
    pnt1 = polygon_unitvecs[:, 1:-1, :]        # shape (n_poly, n_verts-2, 3)
    pnt2 = polygon_unitvecs[:, 2:, :]          # shape (n_poly, n_verts-2, 3)

    u = np.broadcast_to(pnt0, pnt1.shape)      # broadcast pnt0
    areas = _tri_area(u, pnt1, pnt2)      # shape (n_poly, n_verts-2)
    return np.sum(areas, axis=1)               # sum over triangles per polygon


def cell_area_rad(xv_coords, yv_coords):
    """Compute the area of cell(s) using spherical polygons.

    Parameters
    ----------
    xv_coords : np.ndarray
        X-coordinate(s) of cell vertices in degrees.
    yv_coords : np.ndarray
        Y-coordinate(s) of cell vertices in degrees.

    Returns
    -------
    area : np.ndarray
        Area of the cell(s) in square radians.
    """

    # Convert coordinates to radians
    xv_rad = np.deg2rad(xv_coords)
    yv_rad = np.deg2rad(yv_coords)

    # Construct unit vectors
    x = np.cos(yv_rad) * np.cos(xv_rad)
    y = np.cos(yv_rad) * np.sin(xv_rad)
    z = np.sin(yv_rad)

    area = _great_circle_area(np.stack([x, y, z], axis=-1) )
    return area
    