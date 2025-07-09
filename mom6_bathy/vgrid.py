import os
from datetime import datetime
import numpy as np
import xarray as xr

class VGrid:
    """
    Vertical grid class.
    """

    def __init__(
        self,
        dz: np.ndarray,
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None,
        message: str = None,
        author: str = None,
    ):
        self.dz = dz
        self.name = name
        self.repo_root = repo_root or os.getcwd()
        if self.name and save_on_create:
            self._initialize_on_disk(message=message, author=author)

    @property
    def nk(self):
        return len(self.dz)

    @property
    def depth(self):
        return np.sum(self.dz)
    
    @property
    def z(self):
        return np.cumsum(self.dz) - 0.5 * self.dz

    @classmethod
    def sanitize_name(self, name):
        import re
        return re.sub(r'[^A-Za-z0-9_\-]+', '_', name)

    @classmethod
    def uniform(
        cls,
        nk: int,
        depth: float,
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None,
        message: str = None,
        author: str = None,
    ):
        assert nk > 1, "Number of layers must be greater than 1"
        assert depth > 0, "Depth must be greater than 0"
        dz = np.ones(nk) * depth / nk
        dz[-1] = depth - np.sum(dz[:-1])
        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root, message=message, author=author)

    @classmethod
    def hyperbolic(
        cls,
        nk: int,
        depth: float,
        ratio: float,
        name: str = None,
        save_on_create: bool = True,
        repo_root: str = None,
        message: str = None,
        author: str = None,
    ):
        assert nk > 1, "Number of layers must be greater than 1"
        assert ratio > 0, "Ratio must be greater than 0"
        assert depth > 0, "Depth must be greater than 0"
        dz0 = 2 * depth / (nk * (1 + ratio))
        dzbot = ratio * dz0
        dz = dz0 + 0.5 * (
            dzbot - dz0
        ) * (1 + np.tanh(2 * np.pi * (np.arange(nk) / (nk - 1) - 1 / 2)))
        dz[-1] = depth - np.sum(dz[:-1])
        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root, message=message, author=author)

    @classmethod
    def from_file(
        cls,
        filename: str,
        name: str = None,
        save_on_create: bool = False,
        repo_root: str = None
    ):
        assert filename.endswith('.nc'), f"File {filename} is not a NetCDF file"
        assert os.path.exists(filename), f"File {filename} does not exist"
        ds = xr.open_dataset(filename)
        assert 'dz' in ds, f"File {filename} does not contain a 'dz' variable"
        dz = ds['dz'].values
        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root)

    def _get_vgrid_folder(self, root_dir=None, create=True):
        if root_dir is None:
            root_dir = os.getcwd()
        folder = os.path.join(root_dir, "VGrids")
        if create:
            os.makedirs(folder, exist_ok=True)
        return folder

    def _get_nc_path(self, root_dir=None):
        sanitized_name = self.name if self.name is not None else "UnnamedVGrid"
        if root_dir is None:
            root_dir = os.getcwd()
        folder = os.path.join(root_dir, "VGrids")
        nc_path = os.path.join(folder, f"vgrid_{sanitized_name}.nc")
        return nc_path

    def _initialize_on_disk(self, message=None, author=None):
        if not self.name:
            raise ValueError("VGrid must have a name to initialize on disk.")
        nc_path = self._get_nc_path(self.repo_root)
        self.write(nc_path, message=message, author=author)

    def write(self, filename: str, message: str = None, author: str = None):
        dz0 = float(self.dz[0])
        dzbot = float(self.dz[-1])
        ratio = dzbot / dz0 if dz0 != 0 else 1.0
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
        ds.attrs['title'] = self.name or 'Vertical grid for MOM6 simulation'
        ds.attrs['maximum_depth'] = float(self.depth)
        ds.attrs['nk'] = int(self.nk)
        ds.attrs['top_bottom_ratio'] = float(ratio)
        ds.attrs['date_created'] = datetime.now().isoformat()
        if message:
            ds.attrs['message'] = message
        if author:
            ds.attrs['author'] = author
        ds.to_netcdf(filename)