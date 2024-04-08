import os
from datetime import datetime
from typing import Optional
import numpy as np
import xarray as xr
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
        angle T-grid makes with latitude line
    tarea: xr.DataArray
        T-cell area

    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lenx: float,
        leny: float,
        srefine: int = 2,
        xstart: float = 0.0,
        ystart: Optional[float] = None,
        cyclic_x: bool = False,
        tripolar_n: bool = False,
        displace_pole: bool = False,
    ) -> None:
        """
        Grid instance constructor.

        Parameters
        ----------
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        lenx : float
            grid length in x direction, e.g., 360.0 (degrees)
        leny : float
            grid length in y direction, e.g., 160.0 (degrees)
        srefine : int, optional
            refinement factor for the supergrid. 2 by default
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
        """

        # default ystart value (centers the domain at the Equator)
        if ystart is None:
            ystart = -0.5 * leny

        # consistency checks for constructor arguments
        assert nx > 0, "nx must be a positive integer"
        assert ny > 0, "ny must be a positive integer"
        assert np.log2(srefine).is_integer(), "srefine must be a power of two"
        assert (
            not cyclic_x or lenx == 360.0
        ), "cyclic_x is only supported for 360 degree domains."
        assert 0 < lenx <= 360.0, "lenx must be in the range (0, 360]"
        assert 0 < leny <= 180.0, "leny must be in the range (0, 180]"
        assert -90.0 <= ystart <= 90.0, "ystart must be in the range [-90, 90]"
        assert leny + ystart <= 90.0, "leny + ystart must be less than 90"
        assert tripolar_n is False, "tripolar not supported yet"
        assert displace_pole is False, "displaced pole not supported yet"

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
    def is_tripolar(supergrid: xr.Dataset) -> bool:
        """
        Check if a supergrid instance is tripolar.

        Parameters
        ----------
        supergrid : xarray.Dataset
            MOM6 Supergrid dataset
        """
        Grid.check_supergrid(supergrid)

        nx = supergrid.sizes["nx"]

        nlines = (
            0  # number of lines along the top row,
            # (i.e., 2 or more cells with the same x coordinate)
        )
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
    def is_cyclic_x(supergrid: xr.Dataset) -> bool:
        """
        Check if a supergrid instance is cyclic along the x-axis.

        Parameters
        ----------
        supergrid : xarray.Dataset
            MOM6 Supergrid dataset
        """
        Grid.check_supergrid(supergrid)

        return np.allclose(
            (supergrid.x[:, 0] + 360.0) % 360.0,
            (supergrid.x[:, -1] + 360.0) % 360.0,
            rtol=1e-5,
        )

    @classmethod
    def from_supergrid(cls, path: str, srefine: int = 2) -> "Grid":
        """Create a Grid instance from a supergrid file.

        Parameters
        ----------
        path : str
            Path to the supergrid file to be written
        srefine : int, optional
            refinement factor for the supergrid. 2 by default

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

        # create an initial Grid object:
        obj = cls(
            nx=int(len(ds.nx) / srefine),
            ny=int(len(ds.ny) / srefine),
            lenx=(ds.x.max() - ds.x.min()).item(),
            leny=(ds.y.max() - ds.y.min()).item(),
        )

        # override obj.supergrid with the data from the original supergrid file
        obj.supergrid.x = ds.x.data
        obj.supergrid.y = ds.y.data
        obj.supergrid.dx = ds.dx.data
        obj.supergrid.dy = ds.dy.data
        obj.supergrid.area = ds.area.data
        obj.supergrid.angle_dx = ds.angle_dx.data

        return obj

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
        dxCv, dyCu, dxCu, dyCv, angle, and tarea."""

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

        # T point angle:
        self.angle = xr.DataArray(
            sg.angle_dx[1::2, 1::2],
            dims=["ny", "nx"],
            attrs={"name": "angle grid makes with latitude line", "units": "degrees"},
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
