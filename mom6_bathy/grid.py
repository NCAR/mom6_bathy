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
    ) -> None:
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
        
        if nx is not None or ny is not None:
            assert nx is not None and ny is not None, "nx and ny must be provided together"
            assert resolution is None, "resolution cannot be provided with nx and ny"
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
        assert new_name is None or new_name.replace("_", "").isalnum(), "Grid name must be alphanumeric"
        self._name = new_name
    
    def __getitem__(self, slices) -> xr.DataArray:
        """
        Get a subgrid copy based on the provided slices.

        Parameters
        ----------
        slices : tuple
            A tuple of two slices, e.g., [A:B:C, D:E:F]
            The first slice A:B:C corresponds to the j-axis (y-axis) and the second
            slice D:E:F corresponds to the i-axis (x-axis). Examples:
            grid[0:10, 0:20] or grid[:, 0:20] or grid[0:10, :].

        Returns
        -------
        Grid
            A new Grid instance representing the subgrid defined by the slices.
        """

        # Check if args are a tuple of two slices
        assert isinstance(slices, tuple) and len(slices) == 2 and \
            all(isinstance(s, slice) for s in slices), \
            "Must provide both j and i slices when indexing the grid. "\
            "Examples: grid[0:10, 0:20] or grid[:, 0:20] or grid[0:10, :]."

        j_slice, i_slice = slices

        # If both slices are None, return a deep copy of the grid
        if j_slice == slice(None) and i_slice == slice(None):
            return copy.deepcopy(self)

        # Get the slice bounds and steps 
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

        # Negative indices to positive indices
        if j_low < 0:
            j_low = (j_low + self.ny) % self.ny
        if i_low < 0:
            i_low = (i_low + self.nx) % self.nx
        if j_high < 0:
            j_high = (j_high + self.ny) % self.ny
        if i_high < 0:
            i_high = (i_high + self.nx) % self.nx

        # Sanity checks for slice bounds and steps
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

        srefine = 2  # supergrid refinement factor

        # Periodicity checks:
        cyclic_y = self.supergrid.dict["cyclic_y"] and (j_low == 0) and (j_high == self.ny)
        cyclic_x = self.supergrid.dict["cyclic_x"] and (i_low == 0) and (i_high == self.nx)
        tripolar_n = self.supergrid.dict["tripolar_n"] and (i_low == 0) and (i_high == self.nx) and (j_high == self.ny)

        # supergrid slicing:
        s_j_low = j_low * srefine
        s_j_high = (j_high) * srefine + 1
        s_i_low = i_low * srefine
        s_i_high = (i_high) * srefine + 1
            
        # Create a sub-supergrid with the sliced data
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

        # Create a name for the subgrid based on the slices
        # This may (and should) be overriden by the user later
        name = self.name or "subgrid"
        if j_low>0 or j_high < self.ny:
            name += f"_jb{j_low}_je{j_high}"
        if i_low>0 or i_high < self.nx:
            name += f"_ib{i_low}_ie{i_high}"

        # Create a new Grid instance for the subgrid and return it
        sub_grid = Grid(
            nx=int((i_high - i_low) / i_step),
            ny=int((j_high - j_low) / j_step),
            lenx=(sub_supergrid.x.max() - sub_supergrid.x.min()).item(),
            leny=(sub_supergrid.y.max() - sub_supergrid.y.min()).item(),
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
    def from_supergrid(cls, path: str, name: Optional[str] = None) -> "Grid":
        """Create a Grid instance from a supergrid file.

        Parameters
        ----------
        path : str
            Path to the supergrid file to be written
        name : str, optional
            Name of the new grid. If provided, it will be used as the name of the grid.
            If not provided, the name will be derived from the file name.

        Returns
        -------
        Grid
            The Grid instance created from the supergrid file.
        """

        # read supergrid dataset
        ds = xr.open_dataset(path)
        assert (
            ds.x.units == ds.y.units and "degree" in ds.x.units
        ), "Only degrees units are supported in supergrid files"

        # check supergrid
        Grid.check_supergrid(ds)

        srefine = 2  # supergrid refinement factor

        name = name or os.path.basename(path).replace(".nc", "") if os.path.basename(path).endswith(".nc") else os.path.basename(path)

        # create an initial Grid object:
        obj = cls(
            nx=int(len(ds.nx) / srefine),
            ny=int(len(ds.ny) / srefine),
            lenx=(ds.x.max() - ds.x.min()).item(),
            leny=(ds.y.max() - ds.y.min()).item(),
            cyclic_x=Grid.is_cyclic_x(ds),
            name=name
        )

        # override obj.supergrid with the data from the original supergrid file
        obj.supergrid.x = ds.x.data
        obj.supergrid.y = ds.y.data
        obj.supergrid.dx = ds.dx.data
        obj.supergrid.dy = ds.dy.data
        obj.supergrid.area = ds.area.data
        obj.supergrid.angle_dx = ds.angle_dx.data

        # update the MOM6 grid metrics based on the supergrid data
        obj._compute_MOM6_grid_metrics()

        return obj

    @classmethod
    def subgrid_from_supergrid(cls, path: str, llc: tuple[float, float], urc: tuple[float, float], name: str) -> "Grid":
        """Create a Grid instance from a subset of a supergrid file.

        Parameters
        ----------
        path : str
            Path to the full supergrid file to be carved out.
        llc : tuple[float, float]
            Lower left corner coordinates (lat, lon) of the subdomain to extract
        urc : tuple[float, float]
            Upper right corner coordinates (lat, lon) of the subset to extract
        name : str
            Name of the subgrid

        Returns
        -------
        Grid
            The Grid instance created from the supergrid file.
        """

        full_grid = cls.from_supergrid(path)

        assert len(llc) == 2, "llc must be a tuple of two floats"
        assert len(urc) == 2, "urc must be a tuple of two floats"

        # subgrid indices
        llc_j, llc_i = full_grid.get_indices(llc[0], llc[1])
        urc_j, urc_i = full_grid.get_indices(urc[0], urc[1])

        assert llc_j < urc_j, "Lower left corner must be below upper right corner"
        assert llc_i < urc_i, "Lower left corner must be to the left of upper right corner"

        # create a subgrid from the full grid
        subgrid = full_grid[llc_j:urc_j, llc_i:urc_i]
        subgrid.name = name
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
        ds.to_netcdf(path)

    def refine(self, factor: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Super-sample an ocean grid using bilinear interpolation.

        This function refines supergrid coordinates (qlat, qlon)
        to a finer resolution by subdividing each original grid cell into a higher-resolution
        `factor x factor` subgrid, where `factor` is automatically determined to match or exceed
        the target dimensions (`src_nj`, `src_ni`).

        Parameters
        ----------
        factor: int
            The amount by which to make the grid higher resolution

        Returns
        -------
        lat : ndarray of shape (nj, factor, ni, factor)
            Refined latitude field with `factor x factor` points per original cell.
        lon : ndarray of shape (nj, factor, ni, factor)
            Refined longitude field with `factor x factor` points per original cell.

        Notes
        -----
        The output grids can be reshaped to `(nj * factor, ni * factor)` for use in plotting
        or remapping. The interpolation uses bilinear weights based on relative position
        within each original grid cell.
        """

        lon = np.zeros((self.ny, self.nx,factor, factor))
        lat = np.zeros((self.ny, self.nx,factor, factor))
        for j in range(factor):
            ya = (2 * j + 1) / (2 * factor)
            yb = 1.0 - ya
            for i in range(factor):
                xa = (2 * i + 1) / (2 * factor)
                xb = 1.0 - xa
                lon[:,:, j, i] = yb * (
                    xb * self.qlon[:-1, :-1] + xa * self.qlon[:-1, 1:]
                ) + ya * (xb * self.qlon[1:, :-1] + xa * self.qlon[1:, 1:])
                lat[:,:, j, i] = yb * (
                    xb * self.qlat[:-1, :-1] + xa * self.qlat[:-1, 1:]
                ) + ya * (xb * self.qlat[1:, :-1] + xa * self.qlat[1:, 1:])

        return lat, lon

    def coarsen(self, data):
        """
        Coarsen the data on a refined grid by averaging over a factor x factor block.

        Parameters
        ----------
        data: 4D Array
            The data to be coarsened, with shape (nj, factor, ni, factor).

        Returns
        -------
        data : ndarray of shape (nj, ni)
            Coarsened data field
        """

        data = np.nanmean(
            (
                data.reshape(
                    (self.ny, self.nx, data.shape[3] * data.shape[-1])
                )
            ),
            axis=-1,
        )

        return data

