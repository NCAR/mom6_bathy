import os
from datetime import datetime
import numpy as np
import xarray as xr

class VGrid:
    """
    Vertical grid class.
    
    Attributes
    ----------
    dz: np.ndarray
        Array of vertical grid spacings
    nk: int
        Number of vertical levels
    depth: float
        Total depth of the vertical grid
    z: np.ndarray
        Array of layer center depths
    """

    def __init__(
        self,
        dz: np.ndarray,
    ):
        """Create a vertical grid.
        
        Parameters
        ----------
        dz: np.ndarray
            Array of vertical grid spacings (meters)
        """
        self.dz = dz

    @property
    def nk(self):
        """Number of vertical levels."""
        return len(self.dz)

    @property
    def depth(self):
        """Total depth of the water column in meters."""
        return np.sum(self.dz)
    
    @property
    def z(self):
        """Array of vertical grid cell center depths (meters)."""
        return np.cumsum(self.dz) - 0.5 * self.dz

    @classmethod
    def uniform(
        cls, 
        nk: int, 
        depth: float
    ):
        """Create a uniform vertical grid instance.

        Parameters
        ----------
        nk: int
            Number of vertical levels
        depth: float
            Total depth of the water column (meters)
        """

        assert nk > 1, "Number of layers must be greater than 1"
        assert depth > 0, "Depth must be greater than 0"

        # create a uniform vertical grid
        dz = np.ones(nk) * depth / nk

        # update the bottom layer thickness to ensure the total depth is correct
        dz[-1] = depth - np.sum(dz[:-1])

        return cls(dz)


    @classmethod
    def hyperbolic(
        cls,
        nk: int,
        depth: float,
        ratio: float
    ):
        """Create a hyperbolic vertical grid instance. (Adapted from regional-mom6)
        
        Parameters
        ----------
        nk: int
            Number of vertical levels
        ratio: float
            Target ratio of top to bottom layer thicknesses
        depth: float
            Total depth of the water column (meters)
        """

        assert nk > 1, "Number of layers must be greater than 1"
        assert ratio > 0, "Ratio must be greater than 0"
        assert depth > 0, "Depth must be greater than 0"

        dz0 = 2 * depth / (nk * (1 + ratio))
        dzbot = ratio * dz0
        dz = dz0 + 0.5 * (
            dzbot - dz0
        ) * (1 + np.tanh(2 * np.pi * (np.arange(nk) / (nk - 1) - 1 / 2)))

        # update the bottom layer thickness to ensure the total depth is correct
        dz[-1] = depth - np.sum(dz[:-1])

        return cls(dz)


    @classmethod
    def from_file(cls, filename: str):
        """Create a vertical grid from an existing vertical grid file.
        
        Parameters
        ----------
        filename: str
            Name of the NetCDF file containing the vertical grid
        """
    
        assert filename.endswith('.nc'), f"File {filename} is not a NetCDF file"
        assert os.path.exists(filename), f"File {filename} does not exist"
        ds = xr.open_dataset(filename)

        assert 'dz' in ds, f"File {filename} does not contain a 'dz' variable"
        dz = ds['dz'].values

        return cls(dz)

    def write(self, filename: str):
        """Write the vertical grid to a NetCDF file.
        
        Parameters
        ----------
        filename: str
            Name of the NetCDF file to write
        """
        
        ds = xr.Dataset(
            data_vars={
                'dz': (
                    'z', 
                    self.dz, 
                    {
                        'units': 'meter',
                        'long_name': 'nominal thickness of layer',
                        'valid_min': np.min(self.dz),
                        'valid_max': np.max(self.dz),
                    }
                )
            },
        )

        ds.attrs['title'] = f'Vertical grid for MOM6 simulation'
        ds.attrs['maximum_depth'] = self.depth
        ds.attrs['history'] = f'Created on {datetime.now()}'
        ds.to_netcdf(filename)
