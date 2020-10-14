import os, sys
import numpy as np
import xarray as xr
from datetime import datetime
from midas.rectgrid_gen import supergrid

class mom6grid(object):

    """
    Horizontal MOM6 grid. The first step of constructing a MOM6 grid within
    the CESM simpler models framework is to create a mom6grid instance.

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

    def __init__(self, nx, ny, config, axis_units, lenx, leny,
                 srefine=2, xstart=0.0, ystart=0.0, cyclic_x=True, cyclic_y=False,
                 tripolar_n=False, displace_pole=False):
        '''
        mom6grid instance constructor.

        Parameters
        ----------
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        config : str or None
            Grid configuration. Valid values: 'cartesian', 'mercator', 'spherical'
        axis_units : str
            Grid axis units. Valid values: 'degrees', 'm', 'km'
        lenx : float
            grid length in x direction, e.g., 360.0 (degrees)
        leny : float
            grid length in y direction, e.g., 160.0 (degrees)
        srefine : int, optional
            refinement factor for the supergrid. 2 by default
        xstart : float, optional
            starting x coordinate. 0.0 by default.
        ystart : float, optional
            starting y coordinate. 0.0 by default.
        cyclic_x : bool, optional
            flag to make the grid cyclic in x direction. True by default.
        cyclic_y : bool, optional
            flag to make the grid cyclic in y direction. False by default.
        tripolar_n : bool, optional
            flag to make the grid tripolar. False by default.
        displace_pole : bool, optional
            flag to make the grid displaced polar. False by default.
        '''

        # define valid values for certain constructor arguments
        config_valid_vals = ['cartesian', 'mercator', 'spherical']
        axis_units_valid_vals = ['degrees', 'm', 'km']

        # consistency checks for constructor arguments
        assert nx>0, "nx must be a positive integer"
        assert ny>0, "ny must be a positive integer"
        assert config in config_valid_vals, \
            "config value is invalid. pick one: "+" ".join(config_valid_vals)
        assert axis_units in axis_units_valid_vals, \
            "axis_units value is invalid. pick one: "+" ".join(axis_units_valid_vals)
        assert cyclic_y==False, "cyclic_y grids are not supported in MOM6 yet."

        assert tripolar_n==False, "tripolar not supported yet"
        assert displace_pole==False, "displaced pole not supported yet"


        self.supergrid = supergrid(nxtot = nx*srefine,
                                    nytot = ny*srefine,
                                    config = config,
                                    axis_units = axis_units,
                                    ystart = ystart,
                                    leny = leny,
                                    xstart = xstart,
                                    lenx = lenx,
                                    cyclic_x = cyclic_x,
                                    cyclic_y = cyclic_y,
                                    tripolar_n = tripolar_n,
                                    displace_pole = displace_pole
                            )
    @property
    def supergrid(self):
        """MOM6 supergrid contains the grid metrics and the areas at twice the
        nominal resolution of the actual computational grid."""
        return self._supergrid

    @supergrid.setter
    def supergrid(self, new_supergrid):
        print("Updating supergrid...")
        self._supergrid=new_supergrid
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

    @classmethod
    def from_ini(cls, ini_file):
        import configparser
        if isinstance(ini_file, str) and os.path.exists(ini_file): # ini_file is an actual file
            ini = configparser.ConfigParser(ini_file)
        elif isinstance(ini_file, configparser.ConfigParser):
            ini = ini_file
        else: # ini_file is a string in ini format
            ini = configparser.ConfigParser()
            ini.read_string(ini_file)

        g = ini['grid']

        # remove comments from values:
        for option in g:
            g[option] = g[option].split("#")[0].strip()

        # required entries:
        required_grid_options = ['nx', 'ny', 'config', 'axis_units', 'lenx', 'leny']
        assert set(g) >= set(required_grid_options), \
            "Must provide all required grid options: "+",".join(required_grid_options)
        nx = int(g['nx'].strip())
        ny = int(g['ny'].strip())
        config = g['config'].strip()
        axis_units = g['axis_units'].strip()
        lenx = float(g['lenx'].strip())
        leny = float(g['leny'].strip())

        # optional entries:
        srefine = float(g['srefine'].strip()) if 'srefine' in g else 2
        xstart = float(g['xstart'].strip()) if 'xstart' in g else 0.0
        ystart = float(g['ystart'].strip()) if 'ystart' in g else 0.0
        cyclic_x = g['cyclic_x'].strip().lower()=="true" if 'cyclic_x' in g else True
        cyclic_y = g['cyclic_y'].strip().lower()=="true" if 'cyclic_y' in g else False
        tripolar_n = g['tripolar_n'].strip().lower()=="true" if 'tripolar_n' in g else False
        displace_pole = g['displace_pole'].strip().lower()=="true" if 'displace_pole' in g else False

        return cls(nx, ny, config, axis_units, lenx, leny, srefine, xstart, ystart,
                cyclic_x, cyclic_y, tripolar_n, displace_pole)

    def _compute_MOM6_grid_metrics(self):

        sg = self._supergrid
        sg_units = sg.dict['axis_units']
        if sg_units == "m":
            sg_units = "meters"

        # T coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.tlon = xr.DataArray(
            sg.x[1::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"array of t-grid longitudes",
                     "units":units}
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.tlat = xr.DataArray(
            sg.y[1::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"array of t-grid latitudes",
                     "units":units}
        )

        # U coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.ulon = xr.DataArray(
            sg.x[1::2,::2],
            dims = ['ny','nxp'],
            attrs = {"name":"array of u-grid longitudes",
                     "units":units}
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.ulat = xr.DataArray(
            sg.y[1::2,::2],
            dims = ['ny','nxp'],
            attrs = {"name":"array of u-grid latitudes",
                     "units":units}
        )

        # V coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.vlon = xr.DataArray(
            sg.x[::2,1::2],
            dims = ['nyp','nx'],
            attrs = {"name":"array of v-grid longitudes",
                     "units":units}
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.vlat = xr.DataArray(
            sg.y[::2,1::2],
            dims = ['nyp','nx'],
            attrs = {"name":"array of v-grid latitudes",
                     "units":units}
        )

        # Corner coords
        units = "degrees_east" if sg_units == "degrees" else sg_units
        self.qlon = xr.DataArray(
            sg.x[::2,::2],
            dims = ['nyp','nxp'],
            attrs = {"name":"array of q-grid longitudes",
                     "units":units}
        )
        units = "degrees_north" if sg_units == "degrees" else sg_units
        self.qlat = xr.DataArray(
            sg.y[::2,::2],
            dims = ['nyp','nxp'],
            attrs = {"name":"array of q-grid latitudes",
                      "units":units}
        )

        # x-distance between U points, centered at t
        self.dxt = xr.DataArray(
            sg.dx[1::2,::2] + sg.dx[1::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"x-distance between u-points, centered at t",
                     "units":"meters"}
        )
        # y-distance between V points, centered at t
        self.dyt = xr.DataArray(
            sg.dy[::2,1::2] + sg.dy[1::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"y-distance between v-points, centered at t",
                     "units":"meters"}
        )

        # x-distance between q points, centered at v
        self.dxCv = xr.DataArray(
            sg.dx[2::2,::2] + sg.dx[2::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"x-distance between q-points, centered at v",
                     "units":"meters"}
        )

        # y-distance between q points, centered at u
        self.dyCu = xr.DataArray(
            sg.dy[::2,2::2] + sg.dy[1::2,2::2],
            dims = ['ny','nx'],
            attrs = {"name":"y-distance between q-points, centered at u",
                     "units":"meters"}
        )

        # x-distance between y points, centered at u"
        self.dxCu = xr.DataArray(
            sg.dx[1::2,1::2] + np.roll(sg.dx[1::2,1::2], -1, axis=-1),
            dims = ['ny','nx'],
            attrs = {"name":"x-distance between t-points, centered at u",
                     "units":"meters"}
        )

        # y-distance between t points, centered at v"
        self.dyCv = xr.DataArray(
            sg.dy[1::2,1::2] + np.roll(sg.dy[1::2,1::2], -1, axis=0),
            dims = ['ny','nx'],
            attrs = {"name":"y-distance between t-points, centered at v",
                     "units":"meters"}
        )

        # T point angle:
        self.angle = xr.DataArray(
            sg.angle_dx[1::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"angle grid makes with latitude line",
                     "units":"degrees"}
        )

        # T area
        self.tarea = xr.DataArray(
            sg.area[::2,::2] + sg.area[1::2,1::2] + sg.area[::2,1::2] + sg.area[::2,1::2],
            dims = ['ny','nx'],
            attrs = {"name":"area of t-cells",
                     "units":"meters^2"}
        )


    def plot(self, property_name):

        '''
        Plot a given grid property using cartopy.
        Warning: cartopy module must be installed seperately

        Parameters
        ----------
        property_name : str
            The name of the grid property to plot, e.g., 'tlat'.
        '''

        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
        except:
            print("Cannot import the cartopy library, which is required to run this method.")
            return
        import cartopy.feature as cfeature
        import matplotlib.colors as colors
        import cartopy

        if property_name not in self.__dict__:
            print("ERROR: not a valid MOM6 grid property")
            return

        data = self.__dict__[property_name]

        #determine staggering
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

        fig = plt.figure()#figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(central_longitude=-180),cmap='nipy_spectral')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        #ax.coastlines()
        ax.set_global()
        plt.show()



    def plot_cross_section(self, property_name, iy=None, ix=None):

        '''
        Plot the cross-section of a given grid metric.

        Parameters
        ----------
        property_name : str
            The name of the grid property to plot, e.g., 'tlat'.
        iy: int
            y-index of the cross section
        ix: int
            x-inted of the cross section
        '''

        import matplotlib.pyplot as plt

        assert (iy!=None)*(ix!=None)==0, "Cannot provide both iy and ix"

        if property_name not in self.__dict__:
            print("ERROR: not a valid MOM6 grid property")
            return

        data = self.__dict__[property_name]

        #determine staggering
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
            ax.set(xlabel='latitude', ylabel=property_name,
               title=property_name)
            ax.plot(lats[:,iy], data[:,iy])
        elif ix:
            ax.set(xlabel='longitude', ylabel=property_name,
               title=property_name)
            ax.plot(lons[ix,:], data[ix,:])
        ax.grid()

        fig.savefig("test.png")
        plt.show()

    def update_supergrid(self, xdat, ydat):

        '''
        Update the supergrid x and y coordinates. Running this method
        also updates the nominal grid coordinates and metrics.

        Parameters
        ----------
        xdat: np.array
            2-dimensional array of the new x coordinates.
        xdat: np.array
            2-dimensional array of the new y coordinates.
        '''

        new_supergrid = supergrid(
            config = self.supergrid.dict['config'],
            axis_units = self.supergrid.dict['axis_units'],
            xdat = xdat,
            ydat = ydat,
            cyclic_x = self.supergrid.dict['cyclic_x'],
            cyclic_y = self.supergrid.dict['cyclic_y'],
            tripolar_n = self.supergrid.dict['tripolar_n'],
            r0_pole = self.supergrid.dict['r0_pole'],
            lon0_pole = self.supergrid.dict['lon0_pole'],
            doughnut = self.supergrid.dict['doughnut'],
            radius = self.supergrid.dict['radius'],
        )

        self.supergrid = new_supergrid

    def to_netcdf(self, mom6grid_path=None, supergrid_path=None, author=None):

        '''
        Write the horizontal grid and/or supergrid to a netcdf file. The written out netcdf
        supergrid file is to be read in by MOM6 during runtime.

        Parameters
        ----------
        mom6grid_path: str, optional
            Path to the mom6 horizontal grid file to be written.
        supergrid_path: str, optional
            Path to the supergrid file to be written.
        author: str, optional
            Name of the author. If provided, the name will appear in files as metadata.
        '''

        if not (mom6grid_path or supergrid_path):
            raise RuntimeError("Must provide at least one of mom6grid_path and supergrid_path")

        if mom6grid_path:
            raise NotImplementedError()

        if supergrid_path:

            # initialize the dataset:
            ds = xr.Dataset()

            # global attrs:
            ds.attrs['filename'] = os.path.basename(supergrid_path)
            ds.attrs['type'] = "MOM6 supergrid"
            ds.attrs['Created'] = datetime.now().isoformat()
            if author:
                ds.attrs['Author'] = author

            # data arrays:
            ds['y'] = xr.DataArray(self.supergrid.y,
                dims = ['nyp','nxp'],
                attrs = {'units': self.supergrid.dict['axis_units']}
            )
            ds['x'] = xr.DataArray(self.supergrid.x,
                dims = ['nyp','nxp'],
                attrs = {'units': self.supergrid.dict['axis_units']}
            )
            ds['dy'] = xr.DataArray(self.supergrid.dy,
                dims = ['ny','nxp'],
                attrs = {'units': 'meters'}
            )
            ds['dx'] = xr.DataArray(self.supergrid.dx,
                dims = ['nyp','nx'],
                attrs = {'units': 'meters'}
            )
            ds['area'] = xr.DataArray(self.supergrid.area,
                dims = ['ny','nx'],
                attrs = {'units': 'm2'}
            )
            ds['angle_dx'] = xr.DataArray(self.supergrid.angle_dx,
                dims = ['nyp','nxp'],
                attrs = {'units': 'meters'}
            )
            ds.to_netcdf(supergrid_path)
