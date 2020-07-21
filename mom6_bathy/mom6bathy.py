import os, sys
import numpy as np
import xarray as xr
from datetime import datetime
from midas.rectgrid_gen import supergrid
from scipy import interpolate

class mom6bathy(object):
    def __init__(self, grid):
        self._grid = grid
        self._depth = None

    @property
    def depth(self):
        return self._depth.copy()

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




