import os
import json
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
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None
    ):
        """Create a vertical grid.
        
        Parameters
        ----------
        dz: np.ndarray
            Array of vertical grid spacings (meters)
        """
        self.dz = dz
        self.name = name
        self.repo_root = repo_root or os.getcwd()
        if self.name and save_on_create:
            self._initialize_on_disk(message="Initial vgrid creation")


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
    def sanitize_name(self, name):
        import re
        return re.sub(r'[^A-Za-z0-9_\-]+', '_', name)

    def _get_folder_and_paths(self, repo_root=None):
        sanitized_name = self.sanitize_name(self.name) if self.name else "UnnamedVGrid"
        vgrids_dir = os.path.join(repo_root or self.repo_root, "VGrids")
        folder = os.path.join(vgrids_dir, f"vgrid_{sanitized_name}")
        nc_path = os.path.join(folder, f"vgrid_{sanitized_name}.nc")
        json_path = os.path.join(folder, f"vgrid_{sanitized_name}.json")
        return folder, nc_path, json_path

    def _initialize_on_disk(self, message="Initial vgrid creation"):
        folder, nc_path, json_path = self._get_folder_and_paths()
        os.makedirs(folder, exist_ok=True)
        self.write(nc_path)
        self.save_metadata(json_path, message, ncfile=nc_path)

    def save_metadata(self, json_path, message, ncfile=None):
        metadata = {
            "name": self.name,
            "nk": self.nk,
            "depth": float(self.depth),
            "message": message,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(json_path),
            "ncfile": os.path.basename(ncfile) if ncfile else None,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_metadata(cls, json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        return metadata

    @classmethod
    def list_metadata_files(cls, vgrids_dir, filter_checkpoint=True):
        metadata_files = []
        for root, dirs, files in os.walk(vgrids_dir):
            for fname in files:
                if fname.startswith("vgrid_") and fname.endswith(".json"):
                    if filter_checkpoint and "-checkpoint" in fname:
                        continue
                    rel_dir = os.path.relpath(root, vgrids_dir)
                    rel_path = os.path.join(rel_dir, fname) if rel_dir != "." else fname
                    metadata_files.append(rel_path)
        return metadata_files
    
    @classmethod
    def uniform(
        cls,
        nk: int,
        depth: float,
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None
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

        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root)


    @classmethod
    def hyperbolic(
        cls,
        nk: int,
        depth: float,
        ratio: float,
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None
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

        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root)


    @classmethod
    def from_file(
        cls,
        filename: str,
        name: str = None,
        save_on_create: bool = False,
        repo_root: str = None
    ):
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

        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root)

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
