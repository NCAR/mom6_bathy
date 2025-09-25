import os
from datetime import datetime
import numpy as np
import xarray as xr
from typing import Literal

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
        """Create a vertical grid.
        
        Parameters
        ----------
        dz: np.ndarray
            Array of vertical grid spacings (meters)
        """
        
        assert np.all(dz > 0), "Layer thickness cannot be zero or negative."
        
        
        assert np.all(dz > 0), "Layer thickness cannot be zero or negative."
        
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
    
    @property
    def zi(self):
        """Array of vertical grid cell interface depths (meters) - size nk+1.
        Assumes there is a surface interface at 0 meters."""
        return np.insert(np.cumsum(self.dz), 0, 0)
        
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
        repo_root: str = None,
        variable_name: str = "dz",
        variable_type: Literal["layer_thickness", "cell_center", "cell_interface"] = "layer_thickness"
    ):
        """Create a vertical grid from an existing vertical grid file.
        
        Parameters
        ----------
        filename: str
            Name of the NetCDF file containing the vertical grid
        variable_name: str, default: "dz"
            Name of the variable to access within the given NetCDF file to get
            a vertical coordinate.
        variable_type: {"layer_thickness", "cell_center", "cell_interface"}, default: "layer_thickness"
            Type of vertical coordinate information given by the file and variable. 
            
            - layer_thickness: specifies the thickness of each layer in the vertical grid. Size of n for n layers.
            - cell_center: gives the depth at the center of each vertical cell. Size of n for n layers.
            - cell_interface: gives the depth at the interface between each vertical cell. Size of n+1 for n layers.
        """
        assert filename.endswith('.nc'), f"File {filename} is not a NetCDF file"
        assert os.path.exists(filename), f"File {filename} does not exist"
        
        valid_coordinate_types = {"layer_thickness", "cell_center", "cell_interface"}
        assert variable_type in valid_coordinate_types, f"Coordinate type {variable_type} is not a valid option from {valid_coordinate_types}"
        
        
        valid_coordinate_types = {"layer_thickness", "cell_center", "cell_interface"}
        assert variable_type in valid_coordinate_types, f"Coordinate type {variable_type} is not a valid option from {valid_coordinate_types}"
        
        ds = xr.open_dataset(filename)
        assert variable_name in ds, f"File {filename} does not contain a '{variable_name}' variable"
        
        if variable_type == "layer_thickness":
            dz = ds[variable_name].values
        if variable_type == "cell_center":
            dz = _cell_center_to_layer_thickness(ds[variable_name].values)
        if variable_type == "cell_interface":
            dz = _cell_interface_to_layer_thickness(ds[variable_name].values)
            
        assert np.all(dz > 0), "Layer thickness cannot be zero."

        if name is None:
            name = ds.attrs.get("title", None)
            if name is None or name.strip() == "":
                base = os.path.basename(filename)
                if base.startswith("vgrid_") and base.endswith(".nc"):
                    name = base[len("vgrid_"):-3]
                else:
                    name = "UnnamedVGrid"
        return cls(dz, name=name, save_on_create=save_on_create, repo_root=repo_root)

    def _get_vgrid_folder(self, root_dir=None, create=True):
        if root_dir is None:
            root_dir = os.getcwd()
        folder = os.path.join(root_dir, "VGrids")
        if create:
            os.makedirs(folder, exist_ok=True)

            readme_path = os.path.join(folder, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(
                        "# VGrids Directory\n\n"
                        "This folder contains vertical grid definitions generated by "
                        "`mom6_bathy.VGrid`.\n\n"
                        "Each file (`vgrid_*.nc`) includes:\n"
                        "- Layer thickness array (`dz`)\n"
                        "- Metadata such as number of layers (`nk`), total depth, "
                        "and top-to-bottom thickness ratio\n"
                        "- Creation timestamp, and optional commit message/author info\n\n"
                        "These grids are used to configure the vertical discretization "
                        "for MOM6 experiments. Do not edit them manually.\n"
                    )
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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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


def _cell_center_to_layer_thickness(
    cell_centers: np.ndarray
):
    """Convert depth of cell centers to layer thickness.
        
    Parameters
    ----------
    cell_centers: np.ndarray
        Depth of cell centers in meters. Needs to be strictly monotonic.
        We will forcibly correct values to be positive and increasing with index.
    """
    # Check uniform sign of values
    check_sign = np.all(cell_centers > 0) or np.all(cell_centers < 0)
    assert check_sign, "Cell center depths must be all positive or all negative."
    
    # Convert to all positive values, sort, and check monotonicity
    # must be strictly monotonic, meaning no repeating values.
    cell_centers = np.abs(cell_centers)
    
    monotonic = np.all(np.diff(cell_centers) > 0)
    assert monotonic, "Cell center depths must be strictly monotonic."
    
    # put in increasing order
    cell_centers = np.sort(cell_centers)
    
    # Convert from cell centers to layer thickness. 
    temp_data = np.diff(cell_centers, prepend = 0)
    for i in range(0, temp_data.size-1):
        temp_data[i+1] = temp_data[i+1] - temp_data[i]
    
    layer_thickness = temp_data*2
    
    return layer_thickness
    
    
    
def _cell_interface_to_layer_thickness(
    cell_interfaces: np.ndarray
):
    """Convert cell interfaces to layer thickness.
    
    Note: number of cell interfaces = number of layers + 1
        
    Parameters
    ----------
    cell_interfaces: np.ndarray
        Depth of cell interfaces in meters. Needs to be strictly monotonic and surface interface must be at 0 meters. We will forcibly correct values to be positive and increasing with index.
    """
    # Check uniform sign of values
    check_sign = np.all(cell_interfaces >= 0) or np.all(cell_interfaces <= 0)
    assert check_sign, "Cell interface depths must be all positive or all negative (one value, the surface interface, can be 0)."
    
    # Convert to all positive values and check monotonicity
    # must be strictly monotonic, meaning no repeating values.
    cell_interfaces = np.abs(cell_interfaces)
    
    monotonic = np.all(np.diff(cell_interfaces) > 0)
    assert monotonic, "Cell interface depths must be strictly monotonic."
    
    # put in increasing order
    cell_interfaces = np.sort(cell_interfaces)
    
    # Convert cell interface depths to layer thickness
    layer_thickness = np.diff(cell_interfaces)
    
    return layer_thickness