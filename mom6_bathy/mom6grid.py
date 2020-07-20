import os, sys
import numpy as np
import xarray as xr
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__),'./midas'))
from midas.rectgrid_gen import supergrid

class mom6grid(object):

    def __init__(self, nx, ny, config, axis_units, lenx, leny,
                 srefine=2, xstart=0.0, ystart=0.0, cyclic_x=True, cyclic_y=False,
                 tripolar_n=False, displace_pole=False):
        '''
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

    @property
    def supergrid(self):
        return self._supergrid

    @supergrid.setter
    def supergrid(self, new_supergrid):
        print("Updating supergrid...")
        self._supergrid=new_supergrid
        self._supergrid.grid_metrics()
        self._compute_MOM6_grid_metrics()

    @property
    def nx(self):
        return self.tlon.shape[1]

    @property
    def ny(self):
        return self.tlon.shape[0]

    def _compute_MOM6_grid_metrics(self):

        sg = self._supergrid

        # T coords
        self.tlon = xr.DataArray(
                        sg.x[1::2,1::2],
                        dims = ['ny','nx'],
                        attrs = {"name":"array of t-grid longitudes",
                                 "units":"degrees_east"}
                    )
        self.tlat = xr.DataArray(
                        sg.y[1::2,1::2],
                        dims = ['ny','nx'],
                        attrs = {"name":"array of t-grid latitudes",
                                 "units":"degrees_north"}
                    )

        # U coords
        self.ulon = xr.DataArray(
                        sg.x[1::2,::2],
                        dims = ['ny','nxp'],
                        attrs = {"name":"array of u-grid longitudes",
                                 "units":"degrees_east"}
                    )
        self.ulat = xr.DataArray(
                        sg.y[1::2,::2],
                        dims = ['ny','nxp'],
                        attrs = {"name":"array of u-grid latitudes",
                                 "units":"degrees_north"}
                    )

        # V coords
        self.vlon = xr.DataArray(
                        sg.x[::2,1::2],
                        dims = ['nyp','nx'],
                        attrs = {"name":"array of v-grid longitudes",
                                 "units":"degrees_east"}
                    )
        self.vlat = xr.DataArray(
                        sg.y[::2,1::2],
                        dims = ['nyp','nx'],
                        attrs = {"name":"array of v-grid latitudes",
                                 "units":"degrees_north"}
                    )

        # Corner coords
        self.qlon = xr.DataArray(
                        sg.x[::2,::2],
                        dims = ['nyp','nxp'],
                        attrs = {"name":"array of q-grid longitudes",
                                 "units":"degrees_east"}
                    )
        self.qlat = xr.DataArray(
                        sg.y[::2,::2],
                        dims = ['nyp','nxp'],
                        attrs = {"name":"array of q-grid latitudes",
                                  "units":"degrees_north"}
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

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
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

    def to_SCRIP(self, SCRIP_path, title=None):

        ds = xr.Dataset()

        # global attrs:
        ds.attrs['Conventions'] = "SCRIP"
        ds.attrs['date_created'] = datetime.now().isoformat()
        if title:
            ds.attrs['title'] = title

        ds['grid_center_lat'] = xr.DataArray(
            self.tlat.data.flatten(),
            dims = ['grid_size']
        )
        ds['grid_center_lon'] = xr.DataArray(
            self.tlon.data.flatten(),
            dims = ['grid_size']
        )
        ds['grid_corner_lat'] = xr.DataArray(
            np.zeros((ds.dims['grid_size'],4)),
            dims = ['grid_size', 'grid_corners']
        )
        ds['grid_corner_lon'] = xr.DataArray(
            np.zeros((ds.dims['grid_size'],4)),
            dims = ['grid_size', 'grid_corners']
        )
        for i in range(self.nx):
            for j in range(self.ny):
                k = (j*self.nx+i)
                ds['grid_corner_lat'][k,0] = self.qlat[j,i]
                ds['grid_corner_lat'][k,1] = self.qlat[j,i+1]
                ds['grid_corner_lat'][k,2] = self.qlat[j+1,i+1]
                ds['grid_corner_lat'][k,3] = self.qlat[j+1,i]
                ds['grid_corner_lon'][k,0] = self.qlon[j,i]
                ds['grid_corner_lon'][k,1] = self.qlon[j,i+1]
                ds['grid_corner_lon'][k,2] = self.qlon[j+1,i+1]
                ds['grid_corner_lon'][k,3] = self.qlon[j+1,i]
        ds['grid_area'] = xr.DataArray(
            self.tarea.data.flatten(),
            dims = ['grid_size']
        )

        ds.to_netcdf(SCRIP_path)
        



