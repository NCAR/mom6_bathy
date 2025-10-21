import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional


class SupergridBase:
    """Abstract base class defining the MOM6-style supergrid interface."""

    def __init__(self, x, y, dx, dy, area, angle_dx, axis_units):
        """
        Initialize a generic supergrid.

        Parameters
        ----------
        x, y : 2D arrays
            Grid point longitudes and latitudes (or x/y positions).
        dx, dy : 2D arrays
            Cell widths in x and y directions.
        area : 2D array
            Grid cell areas.
        angle : 2D array
            Local grid angle relative to east.
        axis_units : str
            Units of x and y (e.g. "degrees" or "meters").
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.area = area
        self.angle_dx = angle_dx
        self.axis_units = axis_units

    def summary(self):
        """Print a short summary of the grid geometry (shape and dx/dy ranges)."""
        print(
            f"{self.__class__.__name__}: shape={self.x.shape}, "
            f"dx=({self.dx.min()}–{self.dx.max()}), "
            f"dy=({self.dy.min()}–{self.dy.max()})"
        )

    def to_ds(self, author: Optional[str] = None) -> xr.Dataset:
        """
        Export the supergrid to an xarray.Dataset compatible with MOM6.

        Parameters
        ----------
        author : str, optional
            If provided, stored as metadata in the output dataset.
        """
        ds = xr.Dataset()

        # ---- Metadata ----
        ds.attrs["type"] = "MOM6 supergrid"
        ds.attrs["Created"] = datetime.now().isoformat()
        if author:
            ds.attrs["Author"] = author

        # ---- Data variables ----
        ds["y"] = xr.DataArray(
            self.y, dims=["nyp", "nxp"], attrs={"units": self.axis_units}
        )
        ds["x"] = xr.DataArray(
            self.x, dims=["nyp", "nxp"], attrs={"units": self.axis_units}
        )
        ds["dy"] = xr.DataArray(self.dy, dims=["ny", "nxp"], attrs={"units": "meters"})
        ds["dx"] = xr.DataArray(self.dx, dims=["nyp", "nx"], attrs={"units": "meters"})
        ds["area"] = xr.DataArray(self.area, dims=["ny", "nx"], attrs={"units": "m2"})
        ds["angle_dx"] = xr.DataArray(
            self.angle_dx, dims=["nyp", "nxp"], attrs={"units": "radians"}
        )

        return ds


class EqualDegreeSupergrid(SupergridBase):
    """MOM6-style supergrid with constant-degree spacing (lon/lat grid)."""

    @classmethod
    def from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Create a grid from domain extents (lon/lat degrees)."""
        x, y = cls._calc_xy_from_extents(lon_min, len_x, lat_min, len_y, nx, ny)
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(x, y, dx, dy, area, angle_dx, axis_units)

    @classmethod
    def from_xy(cls, x, y):
        """Create a grid directly from coordinate arrays."""
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(x, y, dx, dy, area, angle_dx, axis_units)

    @classmethod
    def _calc_xy_from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Compute full grid geometry for equal-degree spacing."""
        # This builds all geometric quantities (x, y, dx, dy, area, angle)
        # for a supergrid defined in equal-degree (lon/lat) coordinates.

        # ---------------------------------------------------------------------
        # Determine grid resolution and index arrays
        # ---------------------------------------------------------------------
        nx_total = nx * 2  # number of longitudinal cells
        ny_total = ny * 2  # number of latitudinal cells

        jind = np.arange(ny_total)  # latitude cell indices
        iind = np.arange(nx_total)  # longitude cell indices
        jindp = np.arange(ny_total + 1)  # latitude point indices (cell edges)
        iindp = np.arange(nx_total + 1)  # longitude point indices (cell edges)

        # ---------------------------------------------------------------------
        # Compute grid coordinates in degrees
        # ---------------------------------------------------------------------
        grid_y = lat_min + jindp * len_y / ny_total  # latitude edges
        grid_x = lon_min + iindp * len_x / nx_total  # longitude edges

        # Form full 2D coordinate arrays for all cell corners
        x = np.tile(grid_x, (ny_total + 1, 1))
        y = np.tile(grid_y.reshape((ny_total + 1, 1)), (1, nx_total + 1))

        return x, y

    @classmethod
    def _calc_geometry(cls, x, y):
        """Compute full grid geometry for equal-degree spacing."""

        # Update cell counts (used later for shape-dependent arrays)
        nx = x.shape[1] - 1
        ny = x.shape[0] - 1

        # ---------------------------------------------------------------------
        # Compute metric distances on a sphere (approximate)
        # ---------------------------------------------------------------------
        radius = 6.378e6  # Earth radius in meters
        metric = np.deg2rad(radius)  # degrees → meters scaling factor

        # Compute midpoints in each direction
        ymid_j = 0.5 * (y + np.roll(y, shift=-1, axis=0))
        ymid_i = 0.5 * (y + np.roll(y, shift=-1, axis=1))

        # Differences in latitude (dy) and longitude (dx) between adjacent cells
        dy_j = np.roll(y, shift=-1, axis=0) - y
        dy_i = np.roll(y, shift=-1, axis=1) - y
        dx_i = mdist(np.roll(x, shift=-1, axis=1), x)
        dx_j = mdist(np.roll(x, shift=-1, axis=0), x)

        # Compute true distances accounting for spherical geometry
        dx = (
            metric
            * metric
            * (dy_i * dy_i + dx_i * dx_i * np.cos(np.deg2rad(ymid_i)) ** 2)
        )
        dx = np.sqrt(dx)

        dy = (
            metric
            * metric
            * (dy_j * dy_j + dx_j * dx_j * np.cos(np.deg2rad(ymid_j)) ** 2)
        )
        dy = np.sqrt(dy)

        # Trim grid edges for consistency
        dx = dx[:, :-1]
        dy = dy[:-1, :]

        # ---------------------------------------------------------------------
        # Compute cell areas (approximate rectangular areas)
        # ---------------------------------------------------------------------
        area = dx[:-1, :] * dy[:, :-1]

        # ---------------------------------------------------------------------
        # Compute local grid angle relative to east
        # ---------------------------------------------------------------------
        angle_dx = np.zeros((ny + 1, nx + 1))

        # Interior points
        angle_dx[:, 1:-1] = np.arctan2(
            y[:, 2:] - y[:, :-2],
            (x[:, 2:] - x[:, :-2]) * np.cos(np.deg2rad(y[:, 1:-1])),
        )
        # Western boundary
        angle_dx[:, 0] = np.arctan2(
            y[:, 1] - y[:, 0],
            (x[:, 1] - x[:, 0]) * np.cos(np.deg2rad(y[:, 0])),
        )
        # Eastern boundary
        angle_dx[:, -1] = np.arctan2(
            y[:, -1] - y[:, -2],
            (x[:, -1] - x[:, -2]) * np.cos(np.deg2rad(y[:, -1])),
        )

        # Convert angle from degrees to radians
        angle_dx = np.deg2rad(angle_dx)

        # ---------------------------------------------------------------------
        # Record axis units and return all quantities
        # ---------------------------------------------------------------------
        axis_units = "degrees"

        return dx, dy, area, angle_dx, axis_units


class EvenSpacingSupergrid(SupergridBase):
    """MOM6-style supergrid with uniform Cartesian spacing (x/y in meters). Originally by Ashley Barnes in regional_mom6"""

    def __init__(self, lon_min, len_x, lat_min, len_y, resolution):
        x, y, dx, dy, area, angle, axis_units = self._build_grid(
            lon_min, len_x, lat_min, len_y, resolution
        )
        super().__init__(x, y, dx, dy, area, angle, axis_units)

    def _build_grid(self, lon_min, len_x, lat_min, len_y, resolution):
        """Compute full grid geometry for even physical spacing."""
        lon_max = lon_min + len_x
        lat_max = lat_min + len_y

        nx = int(len_x / (resolution / 2))
        if nx % 2 != 1:
            nx += 1

        lons = np.linspace(lon_min, lon_max, nx)  # longitudes in degrees

        # Latitudes evenly spaced by dx * cos(central_latitude)
        central_latitude = np.mean([lat_min, lat_max])  # degrees
        latitudinal_resolution = resolution * np.cos(np.deg2rad(central_latitude))

        ny = int(len_y / (latitudinal_resolution / 2)) + 1

        if ny % 2 != 1:
            ny += 1
        lats = np.linspace(lat_min, lat_max, ny)  # latitudes in degrees

        assert np.all(
            np.diff(lons) > 0
        ), "longitudes array lons must be monotonically increasing"
        assert np.all(
            np.diff(lats) > 0
        ), "latitudes array lats must be monotonically increasing"

        R = 6.371e6  # mean radius of the Earth; https://en.wikipedia.org/wiki/Earth_radius in m

        # compute longitude spacing and ensure that longitudes are uniformly spaced
        dlons = lons[1] - lons[0]

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided array of longitudes must be uniformly spaced"

        # Note: division by 2 because we're on the supergrid
        dx = np.broadcast_to(
            R * np.cos(np.deg2rad(lats)) * np.deg2rad(dlons) / 2,
            (lons.shape[0] - 1, lats.shape[0]),
        ).T

        # dy = R * np.deg2rad(dlats) / 2
        # Note: division by 2 because we're on the supergrid
        dy = np.broadcast_to(
            R * np.deg2rad(np.diff(lats)) / 2, (lons.shape[0], lats.shape[0] - 1)
        ).T

        lon, lat = np.meshgrid(lons, lats)

        area = quadrilateral_areas(lat, lon, R)

        angle_dx = np.zeros_like(lon)

        axis_units = "degrees"
        return lon, lat, dx, dy, area, angle_dx, axis_units


# Helper Functions


def mdist(x1, x2):
    """
    Returns positive distance modulo 360.

    >>> x1=0.0;x2=730.
    >>> d=mdist(x1,x2)
    >>> print(d)
    10.0
    """

    a = np.mod(x1 - x2 + 720.0, 360.0)
    b = np.mod(x2 - x1 + 720.0, 360.0)

    d = np.minimum(a, b)

    return d


def angle_between(v1, v2, v3):
    """Return the angle ``v2``-``v1``-``v3`` (in radians), where
    ``v1``, ``v2``, ``v3`` are 3-vectors. That is, the angle that
    is formed between vectors ``v2 - v1`` and vector ``v3 - v1``.

    Example:

        >>> from regional_mom6.utils import angle_between
        >>> v1 = (0, 0, 1)
        >>> v2 = (1, 0, 0)
        >>> v3 = (0, 1, 0)
        >>> angle_between(v1, v2, v3)
        1.5707963267948966
        >>> from numpy import rad2deg
        >>> rad2deg(angle_between(v1, v2, v3))
        90.0
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    norm_v1xv2 = np.sqrt(vecdot(v1xv2, v1xv2))
    norm_v1xv3 = np.sqrt(vecdot(v1xv3, v1xv3))

    cosangle = vecdot(v1xv2, v1xv3) / (norm_v1xv2 * norm_v1xv3)

    return np.arccos(cosangle)


def vecdot(v1, v2):
    """Return the dot product of vectors ``v1`` and ``v2``.
    ``v1`` and ``v2`` can be either numpy vectors or numpy.ndarrays
    in which case the last dimension is considered the dimension
    over which the dot product is taken.
    """
    return np.sum(v1 * v2, axis=-1)


def latlon_to_cartesian(lat, lon, R=1):
    """Convert latitude and longitude (in degrees) to Cartesian coordinates on
    a sphere of radius ``R``. By default ``R = 1``.

    Arguments:
        lat (float): Latitude (in degrees).
        lon (float): Longitude (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        tuple: Tuple with the Cartesian coordinates ``x, y, z``

    Examples:

        Find the Cartesian coordinates that correspond to point with
        ``(lat, lon) = (0, 0)`` on a sphere with unit radius.

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> latlon_to_cartesian(0, 0)
        (1.0, 0.0, 0.0)

        Now let's do the same on a sphere with Earth's radius

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> R = 6371e3
        >>> latlon_to_cartesian(0, 0, R)
        (6371000.0, 0.0, 0.0)
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Return the area of spherical quadrilaterals on a sphere of radius ``R``.
    By default, ``R = 1``. The quadrilaterals are formed by constant latitude and
    longitude lines on the ``lat``-``lon`` grid provided.

    Arguments:
        lat (numpy.array): Array of latitude points (in degrees).
        lon (numpy.array): Array of longitude points (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        numpy.array: Array with the areas of the quadrilaterals defined by the ``lat``-``lon`` grid
        provided. If the provided ``lat``, ``lon`` arrays are of dimension *m* :math:`\\times` *n*
        then returned areas array is of dimension (*m-1*) :math:`\\times` (*n-1*).

    Example:

        Let's construct a lat-lon grid on the sphere with 60 degree spacing.
        Then we compute the areas of each grid cell and confirm that the
        sum of the areas gives us the total area of the sphere.

        >>> from regional_mom6.utils import quadrilateral_areas
        >>> import numpy as np
        >>> λ = np.linspace(0, 360, 7)
        >>> φ = np.linspace(-90, 90, 4)
        >>> lon, lat = np.meshgrid(λ, φ)
        >>> lon
        array([[  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.]])
        >>> lat
        array([[-90., -90., -90., -90., -90., -90., -90.],
               [-30., -30., -30., -30., -30., -30., -30.],
               [ 30.,  30.,  30.,  30.,  30.,  30.,  30.],
               [ 90.,  90.,  90.,  90.,  90.,  90.,  90.]])
        >>> R = 6371e3
        >>> areas = quadrilateral_areas(lat, lon, R)
        >>> areas
        array([[1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13],
               [4.56284230e+13, 4.56284230e+13, 4.56284230e+13, 4.56284230e+13,
                4.56284230e+13, 4.56284230e+13],
               [1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13]])
        >>> np.isclose(areas.sum(), 4 * np.pi * R**2, atol=np.finfo(areas.dtype).eps)
        True
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def quadrilateral_area(v1, v2, v3, v4):
    """Return the area of a spherical quadrilateral on the unit sphere that
    has vertices on the 3-vectors ``v1``, ``v2``, ``v3``, ``v4``
    (counter-clockwise orientation is implied). The area is computed via
    the excess of the sum of the spherical angles of the quadrilateral from 2π.

    Example:

        Calculate the area that corresponds to half the Northern hemisphere
        of a sphere of radius *R*. This should be 1/4 of the sphere's total area,
        that is π *R*:sup:`2`.

        >>> from regional_mom6.utils import quadrilateral_area, latlon_to_cartesian
        >>> R = 434.3
        >>> v1 = latlon_to_cartesian(0, 0, R)
        >>> v2 = latlon_to_cartesian(0, 90, R)
        >>> v3 = latlon_to_cartesian(90, 0, R)
        >>> v4 = latlon_to_cartesian(0, -90, R)
        >>> quadrilateral_area(v1, v2, v3, v4)
        592556.1793298927
        >>> from numpy import pi
        >>> quadrilateral_area(v1, v2, v3, v4) == pi * R**2
        True
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)

    if not (
        np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v3, v3)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v4, v4)))
    ):
        raise ValueError("vectors provided must have the same length")

    R = np.sqrt(vecdot(v1, v1))

    a1 = angle_between(v1, v2, v4)
    a2 = angle_between(v2, v3, v1)
    a3 = angle_between(v3, v4, v2)
    a4 = angle_between(v4, v1, v3)

    return (a1 + a2 + a3 + a4 - 2 * np.pi) * R**2
