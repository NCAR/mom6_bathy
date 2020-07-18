import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'./midas'))
from midas.rectgrid_gen import supergrid


class mbsupergrid(supergrid):
    
    def __init__(self, nx, ny, config, axis_units, lenx, leny,
                 xstart=0.0, ystart=0.0, cyclic_x=True, cyclic_y=False, 
                 tripolar_n=False, displace_pole=False ):
        '''
        Parameters
        ----------
        nx : int
            Number of supergrid points in x direction
        ny : int
            Number of supergrid points in y direction
        config : str or None
            Grid configuration. Valid values: 'cartesian', 'mercator', 'spherical'
        axis_units : str
            Grid axis units. Valid values: 'degrees', 'm', 'km'
        lenx : float
            grid length in x direction, e.g., 360.0 (degrees)
        leny : float
            grid length in y direction, e.g., 160.0 (degrees)
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
        
        super().__init__(nx, ny, config, axis_units, ystart, leny, xstart, lenx,
                         None, None, cyclic_x, cyclic_y, tripolar_n, displace_pole,
                         #r0_pole=0.0,lon0_pole=0.0,doughnut=0.0,radius=6.378e6
                             file=None # instantiation from file not supported yet
                        )
        
    def plot(self, property_name):
        
        if property_name=="area":
            lons, lats = self.x[:-1,:-1], self.y[:-1,:-1]
        elif property_name=="dx":
            lons, lats = self.x[:,:-1], self.y[:,:-1]
        elif property_name=="dy":
            lons, lats = self.x[:-1,:], self.y[:-1,:]    
        elif property_name=="angle_dx":
            lons, lats = self.x, self.y     
        else:
            print("ERROR: Unknown property")
            return
        
        data = self.__dict__[property_name]
        fig = plt.figure()#figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(central_longitude=-180),cmap='nipy_spectral')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        #ax.coastlines()
        ax.set_global()
        plt.show()

       
