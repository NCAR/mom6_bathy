import os, sys
import numpy as np
import xarray as xr
from datetime import datetime
from midas.rectgrid_gen import supergrid
from scipy import interpolate

class mom6bathy(object):
    def __init__(self, grid, min_depth):
        self._grid = grid
        self._depth = None
        self._min_depth = min_depth

    @property
    def depth(self):
        return self._depth

    @property
    def min_depth(self):
        return self._min_depth

    @property
    def max_depth(self):
        return self.depth.max().item()

    @min_depth.setter
    def min_depth(self, new_min_depth):
        self._min_depth = new_min_depth

    @property
    def tmask(self):
        tmask_da = xr.DataArray(
            np.where(self._depth>=self._min_depth, 1, 0),
            dims = ['ny','nx'],
            attrs = {"name":"T mask"}
        )
        return tmask_da

    def set_flat(self, D):
        self._depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), D),
            dims = ['ny','nx'],
        )

    def set_spoon(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        '''
        Parameters
        ----------
        max_depth : float
            Maximum depth of model in the units of D.
        dedge : float
            The depth [Z ~> m], at the basin edge
        rad_earth : float, optional
            Radius of earth
        expdecay : float, optional
            A decay scale of associated with the sloping boundaries [m]
        '''

        west_lon = self._grid.tlon[0,0]
        south_lat = self._grid.tlat[0,0]
        len_lon = self._grid.supergrid.dict['lenx']
        len_lat = self._grid.supergrid.dict['leny']
        self._depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), max_depth),
            dims = ['ny','nx'],
        )

        D0 = (max_depth - dedge) / \
                ((1.0 - np.exp(-0.5*len_lat*rad_earth*np.pi/(180.0*expdecay))) * \
                 (1.0 - np.exp(-0.5*len_lat*rad_earth*np.pi/(180.0*expdecay))))

        self._depth[:,:] = dedge + D0 * \
            ( np.sin(np.pi * (self._grid.tlon[:,:]-west_lon)/len_lon) * \
             (1.0 - np.exp((self._grid.tlat[:,:] - (south_lat+len_lat))*rad_earth*np.pi / \
                           (180.0*expdecay)) ))


    def set_bowl(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        '''
        Parameters
        ----------
        max_depth : float
            Maximum depth of model in the units of D.
        dedge : float
            The depth [Z ~> m], at the basin edge
        rad_earth : float, optional
            Radius of earth
        expdecay : float, optional
            A decay scale of associated with the sloping boundaries [m]
        '''

        west_lon = self._grid.tlon[0,0]
        south_lat = self._grid.tlat[0,0]
        len_lon = self._grid.supergrid.dict['lenx']
        len_lat = self._grid.supergrid.dict['leny']
        self._depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), max_depth),
            dims = ['ny','nx'],
        )

        D0 = (max_depth - dedge) / \
                ((1.0 - np.exp(-0.5*len_lat*rad_earth*np.pi/(180.0*expdecay))) * \
                 (1.0 - np.exp(-0.5*len_lat*rad_earth*np.pi/(180.0*expdecay))))

        self._depth[:,:] = dedge + D0 * \
            ( np.sin(np.pi * (self._grid.tlon[:,:]-west_lon)/len_lon) * \
             ((1.0 - np.exp(-(self._grid.tlat[:,:]-south_lat)*rad_earth*np.pi / \
                            (180.0*expdecay))) * \
             (1.0 - np.exp((self._grid.tlat[:,:]-(south_lat+len_lat)) * \
                            rad_earth*np.pi/(180.0*expdecay)))))



    def apply_ridge(self, height, width, lon, ilat):
        ridge_lon = [self._grid.tlon[0,0].data,
                     lon-width/2.,
                     lon,
                     lon+width/2.,
                     self._grid.tlon[0,-1].data]
        ridge_height = [0.,0.,-height,0.,0.]
        interp_func = interpolate.interp1d(ridge_lon, ridge_height, kind=2)
        ridge_height_mapped = interp_func(self._grid.tlon[0,:])
        ridge_height_mapped = np.where(ridge_height_mapped <= 0.0, ridge_height_mapped, 0.0)

        for j in range(ilat[0], ilat[1]):
            self._depth[j,:] +=  ridge_height_mapped


    def print_MOM6_runtime_params(self):

        print("{} = {}".format("TRIPOLAR_N", self._grid.supergrid.dict['tripolar_n']))
        print("{} = {}".format("NIGLOBAL", self._grid.nx))
        print("{} = {}".format("NJGLOBAL", self._grid.ny))
        print("{} = {}".format("GRID_CONFIG", "mosaic"))
        print("{} = {}".format("GRID_FILE", "???"))
        print("{} = {}".format("TOPO_CONFIG", "file"))
        print("{} = {}".format("TOPO_FILE", "???"))
        print("{} = {}".format("MAXIMUM_DEPTH", str(self.max_depth)))
        print("{} = {}".format("MINIMUM_DEPTH", str(self.min_depth)))



    def to_topog(self, file_path, title=None):

        ds = xr.Dataset()

        # global attrs:
        ds.attrs['date_created'] = datetime.now().isoformat()
        if title:
            ds.attrs['title'] = title
        else:
            ds.attrs['title'] = "MOM6 topography file"

        ds['depth'] = xr.DataArray(
            self._depth.data,
            dims = ['nj', 'ni'],
            attrs = {'long_name' : 'Depth of ocean bottom',
                     'units' : 'm'}
        )

        ds.to_netcdf(file_path)


    def to_SCRIP(self, SCRIP_path, title=None):

        ds = xr.Dataset()

        # global attrs:
        ds.attrs['Conventions'] = "SCRIP"
        ds.attrs['date_created'] = datetime.now().isoformat()
        if title:
            ds.attrs['title'] = title

        ds['grid_dims'] = xr.DataArray(
            np.array([self._grid.ny, self._grid.nx]).astype(np.int32),
            dims = ['grid_rank']
        )
        ds['grid_center_lat'] = xr.DataArray(
            self._grid.tlat.data.flatten(),
            dims = ['grid_size'],
            attrs = {'units': self._grid.supergrid.dict['axis_units']}
        )
        ds['grid_center_lon'] = xr.DataArray(
            self._grid.tlon.data.flatten(),
            dims = ['grid_size'],
            attrs = {'units': self._grid.supergrid.dict['axis_units']}
        )
        ds['grid_imask'] = xr.DataArray(
            self.tmask.data.astype(np.int32).flatten(),
            dims = ['grid_size'],
            attrs = {'units': "unitless"}
        )

        ds['grid_corner_lat'] = xr.DataArray(
            np.zeros((ds.dims['grid_size'],4)),
            dims = ['grid_size', 'grid_corners'],
            attrs = {'units': self._grid.supergrid.dict['axis_units']}
        )
        ds['grid_corner_lon'] = xr.DataArray(
            np.zeros((ds.dims['grid_size'],4)),
            dims = ['grid_size', 'grid_corners'],
            attrs = {'units': self._grid.supergrid.dict['axis_units']}
        )
        for i in range(self._grid.nx):
            for j in range(self._grid.ny):
                k = (j*self._grid.nx+i)
                ds['grid_corner_lat'][k,0] = self._grid.qlat[j,i]
                ds['grid_corner_lat'][k,1] = self._grid.qlat[j,i+1]
                ds['grid_corner_lat'][k,2] = self._grid.qlat[j+1,i+1]
                ds['grid_corner_lat'][k,3] = self._grid.qlat[j+1,i]
                ds['grid_corner_lon'][k,0] = self._grid.qlon[j,i]
                ds['grid_corner_lon'][k,1] = self._grid.qlon[j,i+1]
                ds['grid_corner_lon'][k,2] = self._grid.qlon[j+1,i+1]
                ds['grid_corner_lon'][k,3] = self._grid.qlon[j+1,i]
        ds['grid_area'] = xr.DataArray(
            self._grid.tarea.data.flatten(),
            dims = ['grid_size']
        )

        ds.to_netcdf(SCRIP_path)




