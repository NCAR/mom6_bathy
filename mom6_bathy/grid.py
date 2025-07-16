import os
import copy
from datetime import datetime
from typing import Optional
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from midas.rectgrid_gen import supergrid as MidasSupergrid


class Grid:
    """
    Horizontal MOM6 grid. The first step of constructing a MOM6 grid within
    the CESM simpler models framework is to create a Grid instance.

    Attributes
    ----------
    tlon: xr.DataArray
        array of t-grid longitudes
    tlat: xr.DataArray
        array of t-grid latitudes
    ulon: xr.DataArray
        array of u-grid longitudes
    ulat: xr.DataArray
        array of u-grid latitudes
    vlon: xr.DataArray
        array of v-grid longitudes
    vlat: xr.DataArray
        array of v-grid latitudes
    qlon: xr.DataArray
        array of corner longitudes
    qlat: xr.DataArray
        array of corner latitudes
    dxt: xr.DataArray
        x-distance between U points, centered at t
    dyt: xr.DataArray
        y-distance between V points, centered at t
    dxCv: xr.DataArray
        x-distance between q points, centered at v
    dyCu: xr.DataArray
        y-distance between q points, centered at u
    dxCu: xr.DataArray
        x-distance between y points, centered at u
    dyCv: xr.DataArray
        y-distance between t points, centered at v
    angle: xr.DataArray
        angle grid makes with latitude line
    angle_q: xr.DataArray
        angle q-grid makes with latitude line
    tarea: xr.DataArray
        T-cell area

    """

    def __init__(
        self,
        lenx: float,
        leny: float,
        nx: int = None,
        ny: int = None,
        resolution: Optional[float] = None,
        xstart: float = 0.0,
        ystart: Optional[float] = None,
        cyclic_x: bool = False,
        tripolar_n: bool = False,
        displace_pole: bool = False,
        name: Optional[str] = None,
        save_on_create: bool = True
    ) -> None:
        self.lenx = lenx
        self.leny = leny
        self.resolution = resolution
        self.xstart = xstart
        """
        Grid instance constructor.

        Parameters
        ----------
        lenx : float
            grid length in x direction, e.g., 360.0 (degrees)
        leny : float
            grid length in y direction, e.g., 160.0 (degrees)
        nx : int, optional
            Number of grid points in x direction
        ny : int, optional
            Number of grid points in y direction
        resolution : float, optional
            grid resolution in degrees. If provided, the grid
            dimensions are computed based on the resolution:
            nx = int(lenx / resolution) and ny = int(leny / resolution)
        xstart : float, optional
            starting x coordinate. 0.0 by default.
        ystart : float, optional
            starting y coordinate. -0.5*leny by default.
        cyclic_x : bool, optional
            flag to make the grid cyclic in x direction. False by default.
        tripolar_n : bool, optional
            flag to make the grid tripolar. False by default.
        displace_pole : bool, optional
            flag to make the grid displaced polar. False by default.
        name : str, optional
            name of the grid. None by default.
        """

        # default ystart value (centers the domain at the Equator)
        if ystart is None:
            ystart = -0.5 * leny
        self.ystart = ystart
        self.name = name
        if nx is not None or ny is not None:
            assert nx is not None and ny is not None, "nx and ny must be provided together"
        else:
            assert resolution is not None, "resolution must be provided if nx and ny are not"
            nx = int(lenx / resolution)
            ny = int(leny / resolution)

        # consistency checks for constructor arguments
        assert nx > 0, "nx must be a positive integer"
        assert ny > 0, "ny must be a positive integer"
        assert (
            not cyclic_x or lenx == 360.0
        ), "cyclic_x is only supported for 360 degree domains."
        assert 0 < lenx <= 360.0, "lenx must be in the range (0, 360]"
        assert 0 < leny <= 180.0, "leny must be in the range (0, 180]"
        assert -90.0 <= ystart <= 90.0, "ystart must be in the range [-90, 90]"
        assert leny + ystart <= 90.0, "leny + ystart must be less than 90"
        assert tripolar_n is False, "tripolar not supported yet"
        assert displace_pole is False, "displaced pole not supported yet"
        self.name = name


        srefine = 2  # supergrid refinement factor

        self.supergrid = MidasSupergrid(
            nxtot=nx * srefine,
            nytot=ny * srefine,
            config="spherical",
            axis_units="degrees",
            ystart=ystart,
            leny=leny,
            xstart=xstart,
            lenx=lenx,
            cyclic_x=cyclic_x,
            cyclic_y=False,  # todo
            tripolar_n=tripolar_n,
            displace_pole=displace_pole,
        )

        # Create the Grids folder for this grid instance
        if self.name and save_on_create:
            self._initialize_on_disk(message="Initial grid creation")

    
    @property
    def name(self) -> str:
        """Name of the grid."""
        return self._name
    
    @property
    def kdtree(self) -> cKDTree:
        """KDTree for fast nearest neighbor search."""
        if not hasattr(self, "_kdtree") or self._kdtree is None:
            self._kdtree = cKDTree(np.column_stack((self.tlat.values.flatten(), self.tlon.values.flatten())))
        return self._kdtree
    
    @name.setter
    def name(self, new_name: str) -> None:
        if new_name is not None:
            sanitized = self.sanitize_name(new_name)
            assert sanitized.replace("_", "").isalnum(), "Grid name must be alphanumeric"
            self._name = sanitized
        else:
            self._name = None
    
    def __getitem__(self, slices, name=None) -> "Grid":
        """
        Get a subgrid copy based on the provided slices.

        Parameters
        ----------
        slices : tuple
            A tuple of two slices, e.g., [A:B:C, D:E:F]
            The first slice A:B:C corresponds to the j-axis (y-axis) and the second
            slice D:E:F corresponds to the i-axis (x-axis). Examples:
            grid[0:10, 0:20] or grid[:, 0:20] or grid[0:10, :].
        name : str, optional
            Name for the subgrid. If not provided, a default name is generated.

        Returns
        -------
        Grid
            A new Grid instance representing the subgrid defined by the slices.
        """

        assert isinstance(slices, tuple) and len(slices) == 2 and \
            all(isinstance(s, slice) for s in slices), \
            "Must provide both j and i slices when indexing the grid. "\
            "Examples: grid[0:10, 0:20] or grid[:, 0:20] or grid[0:10, :]."

        j_slice, i_slice = slices

        if j_slice == slice(None) and i_slice == slice(None):
            return copy.deepcopy(self)

        j_low, j_high, j_step = (
            j_slice.start or 0,
            j_slice.stop or self.ny,
            j_slice.step or 1,
        )
        i_low, i_high, i_step = (
            i_slice.start or 0,
            i_slice.stop or self.nx,
            i_slice.step or 1,
        )

        if j_low < 0:
            j_low = (j_low + self.ny) % self.ny
        if i_low < 0:
            i_low = (i_low + self.nx) % self.nx
        if j_high < 0:
            j_high = (j_high + self.ny) % self.ny
        if i_high < 0:
            i_high = (i_high + self.nx) % self.nx

        assert j_low >= 0, "Lower j slice bound must be non-negative"
        assert i_low >= 0, "Lower i slice bound must be non-negative"
        assert j_step > 0, "j slice step must be positive"
        assert i_step > 0, "i slice step must be positive"
        assert j_low < self.ny, "Lower j slice bound exceeds the grid's ny dimension"
        assert i_low < self.nx, "Lower i slice bound exceeds the grid's nx dimension"
        assert j_high > j_low, "Upper j slice bound must be greater than lower j slice bound"
        assert i_high > i_low, "Upper i slice bound must be greater than lower i slice bound"
        assert j_high <= self.ny, "Upper j slice bound exceeds the grid's ny dimension"
        assert i_high <= self.nx, "Upper i slice bound exceeds the grid's nx dimension"

        srefine = 2

        cyclic_y = self.supergrid.dict["cyclic_y"] and (j_low == 0) and (j_high == self.ny)
        cyclic_x = self.supergrid.dict["cyclic_x"] and (i_low == 0) and (i_high == self.nx)
        tripolar_n = self.supergrid.dict["tripolar_n"] and (i_low == 0) and (i_high == self.nx) and (j_high == self.ny)

        s_j_low = j_low * srefine
        s_j_high = (j_high) * srefine + 1
        s_i_low = i_low * srefine
        s_i_high = (i_high) * srefine + 1

        sub_supergrid = MidasSupergrid(
            config=self.supergrid.dict["config"],
            axis_units=self.supergrid.dict["axis_units"],
            xdat=self.supergrid.x[s_j_low:s_j_high:j_step, s_i_low:s_i_high:i_step],
            ydat=self.supergrid.y[s_j_low:s_j_high:j_step, s_i_low:s_i_high:i_step],
            cyclic_x=cyclic_x,
            cyclic_y=cyclic_y,
            tripolar_n=tripolar_n,
            r0_pole=self.supergrid.dict["r0_pole"],
            lon0_pole=self.supergrid.dict["lon0_pole"],
            doughnut=self.supergrid.dict["doughnut"],
            radius=self.supergrid.dict["radius"],
        )

        try:
            dlon = float(np.abs(sub_supergrid.x[1, 1] - sub_supergrid.x[1, 3])) if sub_supergrid.x.shape[1] > 3 else float(np.abs(sub_supergrid.x[0, 1] - sub_supergrid.x[0, 0]))
            dlat = float(np.abs(sub_supergrid.y[1, 1] - sub_supergrid.y[3, 1])) if sub_supergrid.y.shape[0] > 3 else float(np.abs(sub_supergrid.y[1, 0] - sub_supergrid.y[0, 0]))
            sub_resolution = np.mean([dlon, dlat]) / srefine
        except Exception:
            sub_resolution = None

        # Use provided name or generate default
        if name is None:
            name = self.name or "subgrid"
            if j_low > 0 or j_high < self.ny:
                name += f"_jb{j_low}_je{j_high}"
            if i_low > 0 or i_high < self.nx:
                name += f"_ib{i_low}_ie{i_high}"

        sub_grid = Grid(
            nx=int((i_high - i_low) / i_step),
            ny=int((j_high - j_low) / j_step),
            lenx=(sub_supergrid.x.max() - sub_supergrid.x.min()).item(),
            leny=(sub_supergrid.y.max() - sub_supergrid.y.min()).item(),
            resolution=sub_resolution,
            xstart=float(sub_supergrid.x.min()),
            ystart=float(sub_supergrid.y.min()),
            cyclic_x=cyclic_x,
            name=name
        )
        sub_grid.supergrid = sub_supergrid
        sub_grid._compute_MOM6_grid_metrics()

        return sub_grid

    @staticmethod
    def check_supergrid(supergrid: xr.Dataset) -> None:
        """
        Check if a given supergrid contains the necessary attributes
        and has consistent units.

        Parameters
        ----------
        supergrid : xarray.Dataset
            MOM6 Supergrid dataset
        """
        for attr in ["x", "y", "dx", "dy", "area"]: # todo: add angle_dx
            assert attr in supergrid, f"Cannot find '{attr}' in supergrid dataset."
        assert (
            "units" in supergrid.x.attrs
        ), "units attribute for x coordinate is missing in supergrid dataset."
        assert (
            "units" in supergrid.y.attrs
        ), "units attribute for y coordinate is missing in supergrid dataset."
        assert supergrid.x.units == supergrid.y.units or (
            "degree" in supergrid.x.units and "degree" in supergrid.y.units
        ), "Different units in x and y coordinates not supported"

    @staticmethod
    def is_cyclic_x(supergrid) -> bool:
        """
        Check if a given supergrid x coordinate array is cyclic along the x-axis.

        Parameters
        ----------
        supergrid : xr.DataArray or np.array or MidasSupergrid
            Supergrid to check for cyclic x.
        """
        return np.allclose(
            (supergrid.x[:, 0] + 360.0) % 360.0,
            (supergrid.x[:, -1] + 360.0) % 360.0,
            rtol=1e-5,
        )

    @staticmethod
    def is_tripolar(supergrid) -> bool:
        """Check if the given supergrid x coordinates form a tripolar grid.
        
        Parameters
        ----------
        supergrid : xr.DataArray or np.array or MidasSupergrid
            Supergrid to check if tripolar.
        """

        nlines = (
            0  # number of lines along the top row,
            # (i.e., 2 or more cells with the same x coordinate)
        )

        ny, nx = supergrid.x.shape

        within_line = False
        for i in range(0, nx - 1):
            if not within_line:
                if supergrid.x[-1, i] == supergrid.x[-1, i + 1]:
                    within_line = True
                    nlines += 1
            else:
                if supergrid.x[-1, i] != supergrid.x[-1, i + 1]:
                    within_line = False

        # If there are 3 lines (i.e., 2 or more cells with the same x coordinate),
        # the grid is tripolar
        return nlines == 3
    
    @staticmethod
    def sanitize_name(name):
        import re
        return re.sub(r'[^A-Za-z0-9_]+', '_', name)
    
    def is_rectangular(self, rtol=1e-3) -> bool:
        """Check if the grid is a rectangular lat-lon grid by comparing the
        first and last rows and columns of the tlon and tlat arrays."""

        if (np.allclose(self.tlon[:, 0], self.tlon[0, 0], rtol=rtol) and
            np.allclose(self.tlon[:, -1], self.tlon[0, -1], rtol=rtol) and
            np.allclose(self.tlat[0, :], self.tlat[0, 0], rtol=rtol) and
            np.allclose(self.tlat[-1, :], self.tlat[-1, 0], rtol=rtol)):
            return True
        return False


    @classmethod
    def from_supergrid(cls, path: str, name: Optional[str] = None, save_on_create: bool = False) -> "Grid":
        ds = xr.open_dataset(path)
        assert (
            ds.x.units == ds.y.units and "degree" in ds.x.units
        ), "Only degrees units are supported in supergrid files"

        Grid.check_supergrid(ds)
        srefine = 2

        name = name or os.path.basename(path).replace(".nc", "") if os.path.basename(path).endswith(".nc") else os.path.basename(path)

        # Compute resolution for later assignment
        try:
            dlon = float(np.abs(ds.x[1, 1] - ds.x[1, 3])) if ds.x.shape[1] > 3 else float(np.abs(ds.x[0, 1] - ds.x[0, 0]))
            dlat = float(np.abs(ds.y[1, 1] - ds.y[3, 1])) if ds.y.shape[0] > 3 else float(np.abs(ds.y[1, 0] - ds.y[0, 0]))
            resolution = np.mean([dlon, dlat]) / srefine
        except Exception:
            resolution = None

        obj = cls(
            nx=int(len(ds.nx) / srefine),
            ny=int(len(ds.ny) / srefine),
            lenx=(ds.x.max() - ds.x.min()).item(),
            leny=(ds.y.max() - ds.y.min()).item(),
            resolution=resolution,
            cyclic_x=Grid.is_cyclic_x(ds),
            name=name,
            save_on_create=save_on_create,  # <--- propagate this!
        )

        obj.supergrid.x = ds.x.data
        obj.supergrid.y = ds.y.data
        obj.supergrid.dx = ds.dx.data
        obj.supergrid.dy = ds.dy.data
        obj.supergrid.area = ds.area.data
        obj.supergrid.angle_dx = ds.angle_dx.data

        obj._compute_MOM6_grid_metrics()
        return obj

    @classmethod
    def subgrid_from_supergrid(cls, path: str, llc: tuple[float, float], urc: tuple[float, float], name: str) -> "Grid":
        full_grid = cls.from_supergrid(path, save_on_create=False)  # <--- don't save full grid

        assert len(llc) == 2, "llc must be a tuple of two floats"
        assert len(urc) == 2, "urc must be a tuple of two floats"

        # subgrid indices
        llc_j, llc_i = full_grid.get_indices(llc[0], llc[1])
        urc_j, urc_i = full_grid.get_indices(urc[0], urc[1])

        assert llc_j < urc_j, "Lower left corner must be below upper right corner"
        assert llc_i < urc_i, "Lower left corner must be to the left of upper right corner"

        # Only create the subgrid with the custom name
        subgrid = full_grid.__getitem__((slice(llc_j, urc_j), slice(llc_i, urc_i)), name=name)

        # --- Fix: Set resolution if missing ---
        if getattr(subgrid, "resolution", None) is None:
            # Try to infer from parent grid
            if hasattr(full_grid, "resolution") and full_grid.resolution is not None:
                subgrid.resolution = full_grid.resolution
            else:
                # Fallback: estimate from grid spacing
                dlon = np.abs(subgrid.tlon[0, 1] - subgrid.tlon[0, 0])
                dlat = np.abs(subgrid.tlat[1, 0] - subgrid.tlat[0, 0])
                subgrid.resolution = float(np.mean([dlon, dlat]))
                
        return subgrid

    @property
    def supergrid(self) -> MidasSupergrid:
        """MOM6 supergrid contains the grid metrics and the areas at twice the
        nominal resolution (by default) of the actual computational grid."""
        return self._supergrid

    @supergrid.setter
    def supergrid(self, new_supergrid: MidasSupergrid) -> None:
        assert isinstance(new_supergrid, MidasSupergrid)
        self._supergrid = new_supergrid
        self._supergrid.grid_metrics()
        self._compute_MOM6_grid_metrics()

    @property
    def nx(self):
        """Number of cells in x-direction."""
        return self.tlon.shape[1]

    @property
    def ny(self):
        """Number of cells in y-direction."""
        return self.tlon.shape[0]

    def _compute_MOM6_grid_metrics(self):
        """Compute the MOM6 grid metrics from the supergrid metrics. These 
        include the tlon, tlat, ulon, ulat, vlon, vlat, qlon, qlat, dxt, dyt,
        dxCv, dyCu, dxCu, dyCv, angle, angle_q, and tarea."""

        sg = self._supergrid
        sg_units = sg.dict["axis_units"]
        if sg_units == "m":
            sg_units = "meters"

        # T coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.tlon = xr.DataArray(
            sg.x[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "array of t-grid longitudes", "units": units},
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.tlat = xr.DataArray(
            sg.y[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "array of t-grid latitudes", "units": units},
        )

        # U coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.ulon = xr.DataArray(
            sg.x[1::2, ::2],
            dims=["ny", "nxp"],
            attrs={"name": "array of u-grid longitudes", "units": units},
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.ulat = xr.DataArray(
            sg.y[1::2, ::2],
            dims=["ny", "nxp"],
            attrs={"name": "array of u-grid latitudes", "units": units},
        )

        # V coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.vlon = xr.DataArray(
            sg.x[::2, 1::2],
            dims=["nyp", "nx"],
            attrs={"name": "array of v-grid longitudes", "units": units},
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.vlat = xr.DataArray(
            sg.y[::2, 1::2],
            dims=["nyp", "nx"],
            attrs={"name": "array of v-grid latitudes", "units": units},
        )

        # Corner coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.qlon = xr.DataArray(
            sg.x[::2, ::2],
            dims=["nyp", "nxp"],
            attrs={"name": "array of q-grid longitudes", "units": units},
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.qlat = xr.DataArray(
            sg.y[::2, ::2],
            dims=["nyp", "nxp"],
            attrs={"name": "array of q-grid latitudes", "units": units},
        )

        # x-distance between U points, centered at t
        self.dxt = xr.DataArray(
            sg.dx[1::2, ::2] + sg.dx[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={
                "name": "x-distance between u-points, centered at t",
                "units": "meters",
            },
        )
        # y-distance between V points, centered at t
        self.dyt = xr.DataArray(
            sg.dy[::2, 1::2] + sg.dy[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={
                "name": "y-distance between v-points, centered at t",
                "units": "meters",
            },
        )

        # x-distance between q points, centered at v
        self.dxCv = xr.DataArray(
            sg.dx[2::2, ::2] + sg.dx[2::2, 1::2],
            dims=["ny", "nx"],
            attrs={
                "name": "x-distance between q-points, centered at v",
                "units": "meters",
            },
        )

        # y-distance between q points, centered at u
        self.dyCu = xr.DataArray(
            sg.dy[::2, 2::2] + sg.dy[1::2, 2::2],
            dims=["ny", "nx"],
            attrs={
                "name": "y-distance between q-points, centered at u",
                "units": "meters",
            },
        )

        # x-distance between y points, centered at u"
        self.dxCu = xr.DataArray(
            sg.dx[1::2, 1::2] + np.roll(sg.dx[1::2, 1::2], -1, axis=-1),
            dims=["ny", "nx"],
            attrs={
                "name": "x-distance between t-points, centered at u",
                "units": "meters",
            },
        )

        # y-distance between t points, centered at v"
        self.dyCv = xr.DataArray(
            sg.dy[1::2, 1::2] + np.roll(sg.dy[1::2, 1::2], -1, axis=0),
            dims=["ny", "nx"],
            attrs={
                "name": "y-distance between t-points, centered at v",
                "units": "meters",
            },
        )

        # angle:
        self.angle = xr.DataArray(
            sg.angle_dx[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "angle grid makes with latitude line", "units": "degrees"},
        )

        # q angle
        self.angle_q = xr.DataArray(
            sg.angle_dx[::2, ::2],
            dims=["ny", "nx"],
            attrs={"name": "angle q-grid makes with latitude line", "units": "degrees"},
        )
        # T area
        self.tarea = xr.DataArray(
            sg.area[::2, ::2]
            + sg.area[1::2, 1::2]
            + sg.area[::2, 1::2]
            + sg.area[::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "area of t-cells", "units": "meters^2"},
        )

        # reset _kdtree such that it is recomputed when self.kdtree is accessed again
        self._kdtree = None

    def get_indices(self, tlat: float, tlon: float) -> tuple[int, int]:
        """
        Get the i, j indices of a given tlat and tlon pair.

        Parameters
        ----------
        tlat : float
            The latitude value.
        tlon : float
            The longitude value.

        Returns
        -------
        Tuple[int, int]
            The j, i indices of the given tlat and tlon pair.
        """

        max_tlon = self.tlon.max()
        min_tlon = self.tlon.min()

        # Try to adjust the longitude to the range of the grid (if possible)
        if tlon > max_tlon and (tlon - 360.0) > min_tlon:
            tlon -= 360.0
        elif tlon < min_tlon and (tlon + 360.0) < max_tlon:
            tlon += 360.0

        dist, indices = self.kdtree.query([tlat, tlon])
        j, i = np.unravel_index(indices, self.tlat.shape)
        return int(j), int(i)

    def plot(self, property_name):
        """
        Plot a given grid property using cartopy.
        Warning: cartopy module must be installed seperately

        Parameters
        ----------
        property_name : str
            The name of the grid property to plot, e.g., 'tlat'.
        """

        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
        except ImportError:
            print(
                "Cannot import the cartopy library, which is required to run this method."
            )
            return

        if property_name not in self.__dict__:
            print("ERROR: not a valid MOM6 grid property")
            return

        data = self.__dict__[property_name]

        # determine staggering
        if data.shape == self.tlon.shape:
            lons, lats = self.tlon, self.tlat
        elif data.shape == self.ulon.shape:
            lons, lats = self.ulon, self.ulat
        elif data.shape == self.vlon.shape:
            lons, lats = self.vlon, self.vlat
        elif data.shape == self.qlon.shape:
            lons, lats = self.qlon, self.qlat
        else:
            print("ERROR: cannot determine property staggering")
            return

        data = self.__dict__[property_name]

        fig = plt.figure()  # figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(
            lons,
            lats,
            data,
            transform=ccrs.PlateCarree(central_longitude=-180),
            cmap="nipy_spectral",
        )
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=2,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        # ax.coastlines()
        ax.set_global()
        plt.show()

    def plot_cross_section(self, property_name, iy=None, ix=None):
        """
        Plot the cross-section of a given grid metric.

        Parameters
        ----------
        property_name : str
            The name of the grid property to plot, e.g., 'tlat'.
        iy: int
            y-index of the cross section
        ix: int
            x-inted of the cross section
        """

        import matplotlib.pyplot as plt

        assert (iy is None) or (ix is None), "Cannot provide both iy and ix"

        if property_name not in self.__dict__:
            print("ERROR: not a valid MOM6 grid property")
            return

        data = self.__dict__[property_name]

        # determine staggering
        if data.shape == self.tlon.shape:
            lons, lats = self.tlon, self.tlat
        elif data.shape == self.ulon.shape:
            lons, lats = self.ulon, self.ulat
        elif data.shape == self.vlon.shape:
            lons, lats = self.vlon, self.vlat
        elif data.shape == self.qlon.shape:
            lons, lats = self.qlon, self.qlat
        else:
            print("ERROR: cannot determine property staggering")
            return

        data = self.__dict__[property_name]

        fig, ax = plt.subplots()
        if iy:
            ax.set(xlabel="latitude", ylabel=property_name, title=property_name)
            ax.plot(lats[:, iy], data[:, iy])
        elif ix:
            ax.set(xlabel="longitude", ylabel=property_name, title=property_name)
            ax.plot(lons[ix, :], data[ix, :])
        ax.grid()

        fig.savefig("test.png")
        plt.show()

    def update_supergrid(self, xdat: np.array, ydat: np.array) -> None:
        """
        Update the supergrid x and y coordinates. Running this method
        also updates the nominal grid coordinates and metrics.

        Parameters
        ----------
        xdat: np.array
            2-dimensional array of the new x coordinates.
        ydat: np.array
            2-dimensional array of the new y coordinates.
        """

        new_supergrid = MidasSupergrid(
            config=self._supergrid.dict["config"],
            axis_units=self._supergrid.dict["axis_units"],
            xdat=xdat,
            ydat=ydat,
            cyclic_x=self._supergrid.dict["cyclic_x"],
            cyclic_y=self._supergrid.dict["cyclic_y"],
            tripolar_n=self._supergrid.dict["tripolar_n"],
            r0_pole=self._supergrid.dict["r0_pole"],
            lon0_pole=self._supergrid.dict["lon0_pole"],
            doughnut=self._supergrid.dict["doughnut"],
            radius=self._supergrid.dict["radius"],
        )

        self.supergrid = new_supergrid

    def write_supergrid(
        self, path: Optional[str] = None, author: Optional[str] = None
    ) -> None:
        """
        Write supergrid to a netcdf file. The supergrid file is to be read in by MOM6
        during runtime.

        Parameters
        ----------
        path: str, optional
            Path to the supergrid file to be written.
        author: str, optional
            Name of the author. If provided, the name will appear in files as metadata.
        """

        # initialize the dataset:
        ds = xr.Dataset()

        # global attrs:
        ds.attrs["filename"] = os.path.basename(path)
        ds.attrs["type"] = "MOM6 supergrid"
        ds.attrs["Created"] = datetime.now().isoformat()
        if author:
            ds.attrs["Author"] = author

        # data arrays:
        ds["y"] = xr.DataArray(
            self._supergrid.y,
            dims=["nyp", "nxp"],
            attrs={"units": self._supergrid.dict["axis_units"]},
        )
        ds["x"] = xr.DataArray(
            self._supergrid.x,
            dims=["nyp", "nxp"],
            attrs={"units": self._supergrid.dict["axis_units"]},
        )
        ds["dy"] = xr.DataArray(
            self._supergrid.dy, dims=["ny", "nxp"], attrs={"units": "meters"}
        )
        ds["dx"] = xr.DataArray(
            self._supergrid.dx, dims=["nyp", "nx"], attrs={"units": "meters"}
        )
        ds["area"] = xr.DataArray(
            self._supergrid.area, dims=["ny", "nx"], attrs={"units": "m2"}
        )
        ds["angle_dx"] = xr.DataArray(
            self._supergrid.angle_dx, dims=["nyp", "nxp"], attrs={"units": "meters"}
        )
        ds.attrs["name"] = self.name
        ds.attrs["lenx"] = self.lenx
        ds.attrs["leny"] = self.leny
        ds.attrs["resolution"] = self.resolution
        ds.attrs["xstart"] = self.xstart
        ds.attrs["ystart"] = self.ystart
        
        ds.to_netcdf(path)

    def _initialize_on_disk(self, message="Initial grid creation"):
        """
        Save .nc file for this grid in the Grids directory.
        """
        if not self.name:
            raise ValueError("Grid must have a name to initialize on disk.")

        nc_path = self._get_nc_path()
        if os.path.exists(nc_path):
            return
        self.to_netcdf(nc_path)

    def _get_grid_folder(self, root_dir=None, create=True):
        if root_dir is None:
            root_dir = os.getcwd()
        folder = os.path.join(root_dir, "Grids")
        if create:
            os.makedirs(folder, exist_ok=True)
        return folder

    def _get_nc_path(self, root_dir=None):
        sanitized_name = self.name if self.name is not None else "UnnamedGrid"
        if root_dir is None:
            root_dir = os.getcwd()
        folder = os.path.join(root_dir, "Grids")
        nc_path = os.path.join(folder, f"grid_{sanitized_name}.nc")
        return nc_path

    def to_netcdf(self, path=None, format="grid"):
        if self.resolution is None:
            raise ValueError("Grid resolution is None. Cannot write to NetCDF. Please ensure resolution is set.")
        if path is None:
            path = self._get_nc_path()
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        if format == "supergrid":
            # Write in supergrid format (x, y, dx, dy, area, angle_dx)
            ds = xr.Dataset()
            ds.attrs["filename"] = os.path.basename(path)
            ds.attrs["type"] = "MOM6 supergrid"
            ds.attrs["Created"] = datetime.now().isoformat()
            ds.attrs["name"] = self.name
            ds.attrs["lenx"] = self.lenx
            ds.attrs["leny"] = self.leny
            ds.attrs["resolution"] = self.resolution
            ds.attrs["xstart"] = self.xstart
            ds.attrs["ystart"] = self.ystart

            sg = self._supergrid
            ds["y"] = xr.DataArray(sg.y, dims=["nyp", "nxp"], attrs={"units": sg.dict["axis_units"]})
            ds["x"] = xr.DataArray(sg.x, dims=["nyp", "nxp"], attrs={"units": sg.dict["axis_units"]})
            ds["dy"] = xr.DataArray(sg.dy, dims=["ny", "nxp"], attrs={"units": "meters"})
            ds["dx"] = xr.DataArray(sg.dx, dims=["nyp", "nx"], attrs={"units": "meters"})
            ds["area"] = xr.DataArray(sg.area, dims=["ny", "nx"], attrs={"units": "m2"})
            ds["angle_dx"] = xr.DataArray(sg.angle_dx, dims=["nyp", "nxp"], attrs={"units": "meters"})
            ds.to_netcdf(path)
        else:
            # Default: write in regular grid format (tlon, tlat, etc.)
            ny, nx = self.tlon.shape
            nyp, nxp = self.qlon.shape

            ds = xr.Dataset(
                {
                    "tlon": (["ny", "nx"], self.tlon.values),
                    "tlat": (["ny", "nx"], self.tlat.values),
                    "ulon": (["ny", "nxp"], self.ulon.values),
                    "ulat": (["ny", "nxp"], self.ulat.values),
                    "vlon": (["nyp", "nx"], self.vlon.values),
                    "vlat": (["nyp", "nx"], self.vlat.values),
                    "qlon": (["nyp", "nxp"], self.qlon.values),
                    "qlat": (["nyp", "nxp"], self.qlat.values),
                    "dxt": (["ny", "nx"], self.dxt.values),
                    "dyt": (["ny", "nx"], self.dyt.values),
                    "dxCv": (["ny", "nx"], self.dxCv.values),
                    "dyCu": (["ny", "nx"], self.dyCu.values),
                    "dxCu": (["ny", "nx"], self.dxCu.values),
                    "dyCv": (["ny", "nx"], self.dyCv.values),
                    "angle": (["ny", "nx"], self.angle.values),
                    "angle_q": (["nyp", "nxp"], self.angle_q.values),
                    "tarea": (["ny", "nx"], self.tarea.values),
                },
                coords={
                    "ny": np.arange(ny),
                    "nx": np.arange(nx),
                    "nyp": np.arange(nyp),
                    "nxp": np.arange(nxp),
                }
            )
            ds.attrs.update({
                "name": self.name,
                "lenx": self.lenx,
                "leny": self.leny,
                "resolution": self.resolution,
                "xstart": self.xstart,
                "ystart": self.ystart,
                "nx": self.nx,
                "ny": self.ny,
                "date_created": datetime.now().isoformat(),
            })
            ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path):
        import re
        ds = xr.open_dataset(path)
        # Auto-detect format
        if "tlon" in ds and "tlat" in ds:
            # Standard grid format
            raw_name = ds.attrs.get("name", None)
            if raw_name is not None:
                # Remove ALL leading ocean_hgrid_ prefixes
                name = re.sub(r'^(ocean_hgrid_)+', '', raw_name)
                # Remove ALL trailing _[sessionid] (6 hex digits) segments
                while re.search(r'_[0-9a-f]{6}$', name):
                    name = re.sub(r'_[0-9a-f]{6}$', '', name)
            else:
                name = None
            grid = cls(
                lenx=float(ds.attrs["lenx"]),
                leny=float(ds.attrs["leny"]),
                resolution=float(ds.attrs["resolution"]),
                xstart=float(ds.attrs["xstart"]),
                ystart=float(ds.attrs["ystart"]),
                name=name,
                save_on_create=False,  
            )
            # Assign arrays directly to avoid recomputation
            grid.tlon = ds["tlon"]
            grid.tlat = ds["tlat"]
            if "units" not in grid.tlon.attrs:
                grid.tlon.attrs["units"] = "degrees_east"
            if "units" not in grid.tlat.attrs:
                grid.tlat.attrs["units"] = "degrees_north"
            grid.ulon = ds["ulon"]
            grid.ulat = ds["ulat"]
            grid.vlon = ds["vlon"]
            grid.vlat = ds["vlat"]
            grid.qlon = ds["qlon"]
            grid.qlat = ds["qlat"]
            grid.dxt = ds["dxt"]
            grid.dyt = ds["dyt"]
            grid.dxCv = ds["dxCv"]
            grid.dyCu = ds["dyCu"]
            grid.dxCu = ds["dxCu"]
            grid.dyCv = ds["dyCv"]
            grid.angle = ds["angle"]
            grid.angle_q = ds["angle_q"]
            grid.tarea = ds["tarea"]
            return grid
        elif "x" in ds and "y" in ds:
            # Supergrid format
            return cls.from_supergrid(path)
        else:
            raise ValueError(f"Unrecognized grid file format: {path}")
