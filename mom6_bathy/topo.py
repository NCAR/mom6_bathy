import os
import json
import numpy as np
import xarray as xr
import shutil
from datetime import datetime
from scipy import interpolate
from scipy.ndimage import label
from mom6_bathy.aux import cell_area_rad
from mom6_bathy.git_utils import get_domain_dir, get_repo
from scipy.spatial import cKDTree


class Topo:
    """
    Bathymetry Generator for MOM6 grids (mom6_bathy.grid.Grid).
    """

    def __init__(self, grid, min_depth, snapshot_dir="Topos", save_on_create=True):
        """
        MOM6 Simpler Models bathymetry constructor.

        Parameters
        ----------
        grid: mom6_bathy.grid.Grid
            horizontal grid instance for which the bathymetry is to be created.
        min_depth: float
            Minimum water column depth. Columns with shallow depths are to be masked out.
        """

        self._grid = grid
        self._depth = None
        self._min_depth = min_depth

        # --- Per-Grid Repo Logic ---
        if save_on_create:
            self.SNAPSHOT_DIR = get_domain_dir(self._grid, base_dir=snapshot_dir)
            os.makedirs(self.SNAPSHOT_DIR, exist_ok=True)
            self.repo = get_repo(self.SNAPSHOT_DIR)
            self.repo_root = self.SNAPSHOT_DIR

            self.save_grid_definition(self.SNAPSHOT_DIR)
            self.copy_grid_files_to_snapshot(self.SNAPSHOT_DIR)
        else:
            self.SNAPSHOT_DIR = None
            self.repo = None
            self.repo_root = None

    def save_grid_definition(self, snapshot_dir):
        """Save the associated grid NetCDF path in the topo's snapshot directory."""
        grid_name = self._grid.name or 'grid'
        grid_nc_name = f"grid_{grid_name}.nc"
        grid_nc_path = os.path.join(snapshot_dir, grid_nc_name)
        # Save the grid as NetCDF if not already present
        if not os.path.exists(grid_nc_path):
            self._grid.to_netcdf(grid_nc_path)
        # Save a JSON pointer to the NetCDF file, named with the grid name
        grid_json = os.path.join(snapshot_dir, f"grid_{grid_name}.json")
        with open(grid_json, "w") as f:
            json.dump({"grid_nc": grid_nc_name, "grid_name": grid_name}, f, indent=2)

    def copy_grid_files_to_snapshot(self, snapshot_dir):
        """
        Copy the grid NetCDF and JSON files into the snapshot directory if they exist.
        """
        # Try to get the grid file paths (customize as needed for your project)
        if hasattr(self._grid, "_get_grid_folder_and_path"):
            folder, nc_path, json_path = self._grid._get_grid_folder_and_path(self._grid)
            if os.path.exists(nc_path):
                dest_nc = os.path.join(snapshot_dir, os.path.basename(nc_path))
                if not os.path.exists(dest_nc):
                    shutil.copy2(nc_path, dest_nc)
            if os.path.exists(json_path):
                dest_json = os.path.join(snapshot_dir, os.path.basename(json_path))
                if not os.path.exists(dest_json):
                    shutil.copy2(json_path, dest_json)

    @classmethod
    def load_grid_definition(cls, snapshot_dir, grid_name=None):
        """Load the associated grid from the topo's snapshot directory."""
        from mom6_bathy.grid import Grid
        # If grid_name is not provided, try to find a grid_*.json file
        if grid_name is None:
            files = [f for f in os.listdir(snapshot_dir) if f.startswith("grid_") and f.endswith(".json")]
            if not files:
                raise FileNotFoundError("No grid_*.json file found in snapshot directory.")
            grid_json = os.path.join(snapshot_dir, files[0])
        else:
            grid_json = os.path.join(snapshot_dir, f"grid_{grid_name}.json")
        with open(grid_json, "r") as f:
            meta = json.load(f)
        grid_nc_path = os.path.join(snapshot_dir, meta["grid_nc"])
        return Grid.from_netcdf(grid_nc_path)

    @classmethod
    def from_snapshot(cls, snapshot_dir, min_depth):
        grid = cls.load_grid_definition(snapshot_dir)
        # ... load topo data as before ...
        return cls(grid, min_depth, snapshot_dir=snapshot_dir)

    @classmethod
    def from_domain_dir(cls, domain_dir, default_min_depth=9.5):
        """
        Create a Topo instance from a domain directory.
        Loads grid, min_depth, and original topo if available.
        """
        from mom6_bathy.grid import Grid

        # Find the original grid JSON
        original_files = [
            f for f in os.listdir(domain_dir)
            if f.startswith("original_") and f.endswith(".json") and "min_depth" not in f
        ]
        if not original_files:
            raise FileNotFoundError("No original grid JSON found in domain directory.")
        original_path = os.path.join(domain_dir, original_files[0])
        with open(original_path, "r") as f:
            data = json.load(f)
        domain_id = data.get("domain_id", {})

        grid_kwargs = dict(
            lenx=domain_id.get("lenx"),
            leny=domain_id.get("leny"),
            resolution=domain_id.get("resolution"),
            xstart=domain_id.get("xstart"),
            ystart=domain_id.get("ystart"),
            name=domain_id.get("grid_name")
        )
        grid_kwargs = {k: v for k, v in grid_kwargs.items() if v is not None}
        grid = Grid(**grid_kwargs)
        shape = tuple(domain_id.get("shape", []))
        shape_str = f"{shape[0]}x{shape[1]}"
        snapshot_dir = get_domain_dir(grid)
        os.makedirs(snapshot_dir, exist_ok=True)
        repo = get_repo(snapshot_dir)

        # Load min_depth if available
        original_min_depth_path = os.path.join(snapshot_dir, f"original_min_depth_{domain_id.get('grid_name')}_{shape_str}.json")
        if os.path.exists(original_min_depth_path):
            with open(original_min_depth_path, "r") as f:
                d = json.load(f)
                min_depth = d.get("min_depth", default_min_depth)
        else:
            min_depth = default_min_depth

        topo = cls(grid, min_depth)
        # Load original topo array if available
        original_topo_path = os.path.join(snapshot_dir, f"original_topo_{domain_id.get('grid_name')}_{shape_str}.nc")
        if os.path.exists(original_topo_path):
            ds = xr.open_dataset(original_topo_path)
            topo._depth = ds["depth"]
        return topo
    
    def ensure_original_state(self, snapshot_dir, command_manager, repo, repo_root):
        """
        Ensure that the original (reference) topography and minimum depth files exist for the current grid/domain.
        Also ensures an original snapshot commit and git tracking.
        """
        topo_id = {
            "grid_name": getattr(self._grid, "name", getattr(self._grid, "_name", None)),
            "shape": [int(v) for v in self.depth.data.shape]
        }
        grid_name = topo_id['grid_name']
        shape = topo_id['shape']
        shape_str = f"{shape[0]}x{shape[1]}"
        original_topo_path = os.path.join(snapshot_dir, f"original_topo_{grid_name}_{shape_str}.nc")
        original_min_depth_path = os.path.join(snapshot_dir, f"original_min_depth_{grid_name}_{shape_str}.json")
        original_name = f"original_{grid_name}_{shape_str}"
        original_path = os.path.join(snapshot_dir, f"{original_name}.json")

        if not os.path.exists(original_topo_path):
            ds = xr.Dataset(
                {"depth": (["ny", "nx"], self.depth.data, {"units": "m"})},
                attrs={"date_created": datetime.now().isoformat()}
            )
            ds.to_netcdf(original_topo_path)
        if not os.path.exists(original_min_depth_path):
            with open(original_min_depth_path, "w") as f:
                json.dump({"min_depth": float(self.min_depth)}, f)

        if not os.path.exists(original_path):
            command_manager.save_commit(original_name)

        if not repo.head.is_valid():
            rel_path = os.path.relpath(original_path, repo_root)
            repo.git.add(rel_path)
            repo.index.commit(f"Initial commit: original snapshot {original_name}")

        from mom6_bathy.git_utils import snapshot_action
        snapshot_action('ensure_tracked', repo_root, 
                        file_path=original_path, commit_msg=f"Update original snapshot {original_name}")
    
    def get_domain_id(self):
        grid = self._grid
        grid_name = getattr(grid, "name", getattr(grid, "_name", None))
        shape = [int(v) for v in self.depth.data.shape]
        lenx = getattr(grid, "lenx", None)
        leny = getattr(grid, "leny", None)
        resolution = getattr(grid, "resolution", None)
        xstart = getattr(grid, "xstart", None)
        ystart = getattr(grid, "ystart", None)
        return {
            "grid_name": grid_name,
            "shape": shape,
            "lenx": lenx,
            "leny": leny,
            "resolution": resolution,
            "xstart": xstart,
            "ystart": ystart,
        }
    
    def get_domain_options(self, snapshot_dir):
        """Return a list of (label, value) tuples for available domains."""
        from mom6_bathy.git_utils import list_domain_dirs
        base_dir = os.path.dirname(snapshot_dir)
        domains = list_domain_dirs(base_dir)
        options = []
        for d in domains:
            label = d.replace("domain_", "")
            options.append((label, d))
        return options

    def get_current_domain(self, domain_options, snapshot_dir):
        """Return the value of the current domain if present in options, else None."""
        current = None
        for label, value in domain_options:
            if value == os.path.basename(snapshot_dir):
                current = value
                break
        return current

    def persist_last_domain(self, snapshot_dir, domain_id=None, snapshot_name=None, load=False):
        """Save or load the last used domain and snapshot to/from a JSON file."""
        path = os.path.join(snapshot_dir, ".last_domain.json")
        if load:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return data.get("domain_id"), data.get("snapshot_name")
            except Exception:
                return None, None
        else:
            try:
                with open(path, "w") as f:
                    json.dump({"domain_id": domain_id, "snapshot_name": snapshot_name}, f)
            except Exception:
                pass

    def check_restore_last_domain(self, snapshot_dir, get_topo_id, restore_last=True):
        """Restore last domain/grid only if it matches the current topo's grid_name and shape."""
        if not restore_last:
            return None, None
        last_domain_id, snapshot_name = self.persist_last_domain(snapshot_dir, load=True)
        if last_domain_id is not None:
            current_id = get_topo_id()
            if (last_domain_id.get("grid_name") == current_id.get("grid_name") and
                last_domain_id.get("shape") == current_id.get("shape")):
                new_topo, snap = self.restore_last_domain(snapshot_dir, get_topo_id)
                return new_topo, snap
        return None, None

    def restore_last_domain(self, snapshot_dir, get_topo_id):
        """Restore the last used domain/grid and snapshot, if available. Returns snapshot_name if present, else None."""
        domain_id, snapshot_name = self.persist_last_domain(snapshot_dir, load=True)
        if not domain_id:
            return None  # Nothing to restore

        try:
            from mom6_bathy.grid import Grid
            from mom6_bathy.topo import Topo
            grid_kwargs = {k: v for k, v in dict(
                lenx=domain_id.get("lenx"),
                leny=domain_id.get("leny"),
                resolution=domain_id.get("resolution"),
                xstart=domain_id.get("xstart"),
                ystart=domain_id.get("ystart"),
                name=domain_id.get("grid_name")
            ).items() if v is not None}
            new_grid = Grid(**grid_kwargs)
            min_depth = domain_id.get("min_depth", 9.5)
            shape = tuple(domain_id.get("shape", []))
            shape_str = f"{shape[0]}x{shape[1]}"
            original_min_depth_path = os.path.join(snapshot_dir, f"original_min_depth_{domain_id.get('grid_name')}_{shape_str}.json")
            if os.path.exists(original_min_depth_path):
                with open(original_min_depth_path, "r") as f:
                    d = json.load(f)
                    min_depth = d.get("min_depth", min_depth)
            new_topo = Topo(new_grid, min_depth)
            original_topo_path = os.path.join(snapshot_dir, f"original_topo_{domain_id.get('grid_name')}_{shape_str}.nc")
            if os.path.exists(original_topo_path):
                ds = xr.open_dataset(original_topo_path)
                new_topo._depth = ds["depth"]
            # Set state in the editor after calling this!
            return new_topo, snapshot_name  # Return the new topo and snapshot name
        except Exception as e:
            print(f"[WARN] Could not restore last domain: {e}")
            return None, None
    
    def apply_edit(self, cmd):
        self.command_manager.execute(cmd)
        self.command_manager.save_commit("_autosave_working")

    def undo_last_edit(self):
        self.command_manager.undo()

    def redo_last_edit(self):
        self.command_manager.redo()

    def to_snapshot_dict(self):
        return {
            "domain_id": self.get_domain_id(),
            "depth": self._depth.data.tolist(),
            "min_depth": self._min_depth,
            # Add more fields as needed
        }

    def save_snapshot(self, path, extra_metadata=None):
        data = self.to_snapshot_dict()
        if extra_metadata:
            data.update(extra_metadata)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_snapshot_file(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        # Reconstruct grid and topo
        from mom6_bathy.grid import Grid
        domain_id = data["domain_id"]
        grid_kwargs = {k: v for k, v in dict(
            lenx=domain_id.get("lenx"),
            leny=domain_id.get("leny"),
            resolution=domain_id.get("resolution"),
            xstart=domain_id.get("xstart"),
            ystart=domain_id.get("ystart"),
            name=domain_id.get("grid_name")
        ).items() if v is not None}
        grid = Grid(**grid_kwargs)
        topo = cls(grid, data.get("min_depth", 9.5))
        topo._depth = xr.DataArray(
            np.array(data["depth"]),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )
        return topo

    @classmethod
    def from_topo_file(cls, grid, topo_file_path, min_depth=0.0):
        """
        Create a bathymetry object from an existing topog file.

        Parameters
        ----------
        grid: mom6_bathy.grid.Grid
            horizontal grid instance for which the bathymetry is to be created.
        topo_file_path: str
            Path to an existing MOM6 topog file.
        min_depth: float, optional
            Minimum water column depth (m). Columns with shallower depths are to be masked out.
        """

        topo = cls(grid, 0.0, save_on_create=False)
        topo.set_depth_via_topog_file(topo_file_path)
        topo.min_depth = min_depth
        return topo

    @property
    def depth(self):
        """
        MOM6 grid depth array (m). Positive below MSL.
        """
        return self._depth

    @depth.setter
    def depth(self, depth):
        """
        Apply a custom bathymetry via a user-defined depth array.

        Parameters
        ----------
        depth: np.array
            2-D Array of ocean depth (m).
        """

        if np.isscalar(depth):
            self.set_flat(depth)
            return

        assert depth.shape == (
            self._grid.ny,
            self._grid.nx,
        ), "Incompatible depth array shape"

        if isinstance(depth, xr.DataArray):
            depth = depth.data
        else:
            assert isinstance(
                depth, np.ndarray
            ), "depth must be a numpy array or xarray DataArray"

        self._depth = xr.DataArray(
            depth,
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

    @property
    def min_depth(self):
        """
        Minimum water column depth. Columns with shallow depths are to be masked out.
        """
        return self._min_depth

    @property
    def max_depth(self):
        """
        Maximum water column depth.
        """
        return self.depth.max().item()

    @min_depth.setter
    def min_depth(self, new_min_depth):
        self._min_depth = new_min_depth

    @property
    def tmask(self):
        """
        Ocean domain mask at T grid. 1 if ocean, 0 if land.
        """
        tmask_da = xr.DataArray(
            np.where(self._depth > self._min_depth, 1, 0),
            dims=["ny", "nx"],
            attrs={"name": "T mask"},
        )
        return tmask_da
    
    @property
    def umask(self):
        """
        Ocean domain mask on U grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        umask = xr.DataArray(
            np.ones(self._grid.ulat.shape, dtype=int),
            dims = ['yh','xq'],
            attrs={"name": "U mask"})
        
        # Fill umask with mask values
        umask[:,:-1] &= tmask.values # h-point translates to the left u-point
        umask[:,1:] &= tmask.values # h-point translates to the right u-point

        return umask
    
    @property
    def vmask(self):
        """
        Ocean domain mask on V grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        vmask = xr.DataArray(
            np.ones(self._grid.vlat.shape, dtype=int),
            dims = ['yq','xh'],
            attrs={"name": "V mask"})
        
        # Fill vmask with mask values
        vmask[:-1,:] &= tmask.values # h-point translates to the bottom v-point
        vmask[1:,:] &= tmask.values # h-point translates to the top v-point

        return vmask
    
    @property
    def qmask(self):
        """
        Ocean domain mask on Q grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        qmask = xr.DataArray(
            np.ones(self._grid.qlat.shape, dtype=int),
            dims = ['yq','xq'],
            attrs={"name": "Q mask"})
        
        # Fill qmask with mask values
        qmask[:-1, :-1] &= tmask.values    # top-left of h goes to top-left q
        qmask[:-1, 1:]  &= tmask.values     # top-right
        qmask[1:, :-1]  &= tmask.values   # bottom-left
        qmask[1:, 1:]   &= tmask.values     # bottom-right 

        # Corners of the qmask are always land -> regional cases
        qmask[0, 0] = 0
        qmask[0, -1] = 0
        qmask[-1, 0] = 0
        qmask[-1, -1] = 0

        return qmask
          

        
    @property
    def basintmask(self):
        """
        Ocean domain mask at T grid. Seperate number for each connected water cell, 0 if land.
        """
        res, num_features = label(self.tmask)
        
        return xr.DataArray(res)
    
    @property
    def supergridmask(self):
        """
        Ocean domain mask on supergrid. 1 if ocean, 0 if land.
        """

        supergridmask = xr.DataArray(
            np.zeros(self._grid._supergrid.x.shape, dtype=int),
            dims=["nyp", "nxp"],
            attrs={"name": "supergrid mask"})
        supergridmask[::2, ::2] = self.qmask.values
        supergridmask[::2, 1::2] = self.vmask.values
        supergridmask[ 1::2,::2] = self.umask.values
        supergridmask[ 1::2,1::2] = self.tmask.values
        return supergridmask

    def point_is_ocean(self, lons,lats):
        """
        Given a list of coordinates, return a list of booleans indicating if the coordinates are in the ocean (True) or land (False)
        """
        assert len(lons) == len(lats), "Lons & Lats must be the same length, they describe a set of points"

        is_ocean=[]
        for i in range(len(lons)):
            match = np.where((self._grid._supergrid.x == lons[i]) & (self._grid._supergrid.y == lats[i]))
            is_ocean.append(self.supergridmask[match[0],match[1]].item())
        return is_ocean

    def set_flat(self, D):
        """
        Create a flat bottom bathymetry with a given depth D.

        Parameters
        ----------
        D: float
            Bathymetric depth of the flat bottom to be generated.
        """
        self._depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), D),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

    def set_depth_via_topog_file(self, topog_file_path):
        """
        Apply a bathymetry read from an existing topog file

        Parameters
        ----------
        topog_file_path: str
            absolute path to an existing MOM6 topog file
        """

        assert os.path.exists(
            topog_file_path
        ), f"Cannot find topog file at {topog_file_path}."

        ds_topo = xr.open_dataset(topog_file_path)
        assert "depth" in ds_topo, f"Cannot find the 'depth' field in topog file {topog_file_path}"
        depth = ds_topo["depth"]

        if depth.shape[0] < self._grid.ny or depth.shape[1] < self._grid.nx:
            raise ValueError(
                f"Topography data in {topog_file_path} is smaller than the grid size "
                f"({depth.shape[0]}x{depth.shape[1]} < {self._grid.ny}x{self._grid.nx}). "
            )
        elif depth.shape[0] > self._grid.ny or depth.shape[1] > self._grid.nx:
            assert (
                'geolat' in ds_topo and 'geolon' in ds_topo
            ), f"Topog file {topog_file_path} does not contain geolat and geolon fields, "
            "which are required to determine if the grid is a subgrid of the topog file, "
            "since the topography data is larger than the grid (in index space). "

            # Determine if the grid is a subgrid of the topog file
            geolat = ds_topo['geolat']
            geolon = ds_topo['geolon']

            # find the closest cell in the topog file to the (sub)grid's origin (southwest corner)
            topog_kdtree =  cKDTree(
                np.column_stack((geolat.data.flatten(), geolon.data.flatten()))
            )
            _, indices = topog_kdtree.query(
                [self._grid.tlat[0, 0].item(), self._grid.tlon[0, 0].item()]
            )
            cj, ci = np.unravel_index(indices, geolon.shape)

            assert 0 <= cj < geolat.shape[0] - self._grid.ny, (
                f"Topography data in {topog_file_path} appears to only contain a subregion "
                f"of the grid, and does not contain enough rows to accommodate the grid size "
                f"({self._grid.ny}). "
            )
            assert 0 <= ci < geolon.shape[1] - self._grid.nx, (
                f"Topography data in {topog_file_path} appears to only contain a subregion "
                f"of the grid, and does not contain enough columns to accommodate the grid size "
                f"({self._grid.nx}). "
            )

            # Compare the coords of grid with the coords of the subregion of the topog
            # data where it may overlap with the grid
            grid_overlaps_topo = (
                np.all(
                    np.isclose(
                        geolat[cj:cj + self._grid.ny, ci:ci + self._grid.nx],
                        self._grid.tlat.data,
                        rtol=1e-5
                    )
                )
                and np.all(
                    np.isclose(
                        geolon[cj:cj + self._grid.ny, ci:ci + self._grid.nx],
                        self._grid.tlon.data,
                        rtol=1e-5
                    )
                )
            )
            if not grid_overlaps_topo:
                raise ValueError(
                    f"The topography data in {topog_file_path} is larger than the grid "
                    f"data which does not appear to be a subgrid of the topography data. "
                    f"Topography data shape: {depth.shape}, grid shape: "
                    f"({self._grid.ny}, {self._grid.nx}). "
                )

            # If the grid is a subgrid of the topog data, extract the subregion
            depth = depth[cj:cj + self._grid.ny, ci:ci + self._grid.nx]
        
        else:
            pass # the depth array is the right size

        self.depth = depth

    def set_spoon(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        """
        Create a spoon-shaped bathymetry. Same effect as setting the TOPO_CONFIG
        parameter to "spoon".

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
        """

        west_lon = self._grid.tlon[0, 0]
        south_lat = self._grid.tlat[0, 0]
        nx = self._grid.nx
        ny = self._grid.ny
        lenx = self._grid.supergrid.dict["lenx"]
        leny = self._grid.supergrid.dict["leny"]
        self._depth = xr.DataArray(
            np.full((ny, nx), max_depth),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

        D0 = (max_depth - dedge) / (
            (1.0 - np.exp(-0.5 * leny * rad_earth * np.pi / (180.0 * expdecay)))
            * (1.0 - np.exp(-0.5 * leny * rad_earth * np.pi / (180.0 * expdecay)))
        )

        self._depth[:, :] = dedge + D0 * (
            np.sin(np.pi * (self._grid.tlon[:, :] - west_lon) / lenx)
            * (
                1.0
                - np.exp(
                    (self._grid.tlat[:, :] - (south_lat + leny))
                    * rad_earth
                    * np.pi
                    / (180.0 * expdecay)
                )
            )
        )

    def set_bowl(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        """
        Create a bowl-shaped bathymetry. Same effect as setting the TOPO_CONFIG parameter to "bowl".

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
        """

        west_lon = self._grid.tlon[0, 0]
        south_lat = self._grid.tlat[0, 0]
        len_lon = self._grid.supergrid.dict["lenx"]
        len_lat = self._grid.supergrid.dict["leny"]
        self._depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), max_depth),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

        D0 = (max_depth - dedge) / (
            (1.0 - np.exp(-0.5 * len_lat * rad_earth * np.pi / (180.0 * expdecay)))
            * (1.0 - np.exp(-0.5 * len_lat * rad_earth * np.pi / (180.0 * expdecay)))
        )

        self._depth[:, :] = dedge + D0 * (
            np.sin(np.pi * (self._grid.tlon[:, :] - west_lon) / len_lon)
            * (
                (
                    1.0
                    - np.exp(
                        -(self._grid.tlat[:, :] - south_lat)
                        * rad_earth
                        * np.pi
                        / (180.0 * expdecay)
                    )
                )
                * (
                    1.0
                    - np.exp(
                        (self._grid.tlat[:, :] - (south_lat + len_lat))
                        * rad_earth
                        * np.pi
                        / (180.0 * expdecay)
                    )
                )
            )
        )

    def apply_ridge(self, height, width, lon, ilat):
        """
        Apply a ridge to the bathymetry.

        Parameters
        ----------
        height : float
            Height of the ridge to be added.
        width : float
            Width of the ridge to be added.
        lon : float
            Longitude where the ridge is to be centered.
        ilat : pair of integers
            Initial and final latitude indices for the ridge.
        """

        ridge_lon = [
            self._grid.tlon[0, 0].data,
            lon - width / 2.0,
            lon,
            lon + width / 2.0,
            self._grid.tlon[0, -1].data,
        ]
        ridge_height = [0.0, 0.0, -height, 0.0, 0.0]
        interp_func = interpolate.interp1d(ridge_lon, ridge_height, kind=2)
        ridge_height_mapped = interp_func(self._grid.tlon[0, :])
        ridge_height_mapped = np.where(
            ridge_height_mapped <= 0.0, ridge_height_mapped, 0.0
        )

        for j in range(ilat[0], ilat[1]):
            self._depth[j, :] += ridge_height_mapped

    def apply_land_frac(
        self,
        landfrac_filepath,
        landfrac_name,
        xcoord_name,
        ycoord_name,
        depth_fillval=0.0,
        cutoff_frac=0.5,
        method="bilinear",
    ):
        """
        Given a dataset containing land fraction, generate and apply ocean mask.

        Parameters
        ----------
        landfrac_filepath : str
            Path the netcdf file containing the land fraction field.
        landfrac_name : str
            The field name corresponding to the land fraction  (e.g., "landfrac").
        xcoord_name : str
            The name of the x coordinate of the landfrac dataset (e.g., "lon").
        ycoord_name : str
            The name of the y coordinate of the landfrac dataset (e.g., "lat").
        depth_fillval : float
            The depth value for dry cells.
        cutoff_frac : float
            Cells with landfrac > cutoff_frac are deemed land cells.
        method : str
            Mapping method for determining the ocean mask (lnd -> ocn)
        """

        import xesmf as xe

        assert isinstance(landfrac_filepath, str), "landfrac_filepath must be a string"
        assert landfrac_filepath.endswith(
            ".nc"
        ), "landfrac_filepath must point to a netcdf file"
        ds = xr.open_dataset(landfrac_filepath)

        assert isinstance(landfrac_name, str), "landfrac_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {landfrac_name} in {landfrac_filepath}"
        assert isinstance(xcoord_name, str), "xcoord_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {xcoord_name} in {landfrac_filepath}"
        assert isinstance(ycoord_name, str), "ycoord_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {ycoord_name} in {landfrac_filepath}"
        assert isinstance(
            depth_fillval, float
        ), f"depth_fillval={depth_fillval} must be a float"
        assert (
            depth_fillval < self._min_depth
        ), f"depth_fillval (the depth of dry cells) must be smaller than the minimum depth {self._min_depth}"
        assert isinstance(
            cutoff_frac, float
        ), f"cutoff_frac={cutoff_frac} must be a float"
        assert (
            0.0 <= cutoff_frac <= 1.0
        ), f"cutoff_frac={cutoff_frac} must be 0<= and <=1"

        valid_methods = [
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        ]
        assert (
            method in valid_methods
        ), f"{method} is not a valid mapping method. Choose from: {valid_methods}"

        ds_mapped = xr.Dataset(
            data_vars={}, coords={"lat": self._grid.tlat, "lon": self._grid.tlon}
        )

        regridder = xe.Regridder(
            ds, ds_mapped, method, periodic=self._grid.supergrid.dict["cyclic_x"]
        )
        mask_mapped = regridder(ds.landfrac)
        self._depth.data = np.where(
            mask_mapped > cutoff_frac, depth_fillval, self._depth
        )

    def write_topo(self, file_path, title=None):
        """
        Write the TOPO_FILE (bathymetry file) in netcdf format. The written file is
        to be read in by MOM6 during runtime.

        Parameters
        ----------
        file_path: str
            Path to TOPO_FILE to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title
        else:
            ds.attrs["title"] = "MOM6 topography file"
        ds.attrs["min_depth"] = self.min_depth
        ds.attrs["max_depth"] = self.max_depth

        ds["y"] = xr.DataArray(
            self._grid.tlat,
            dims=["ny", "nx"],
            attrs={
                "long_name": "array of t-grid latitudes",
                "units": self._grid.tlat.attrs.get("units", "degrees_north"),
            },
        )

        ds["x"] = xr.DataArray(
            self._grid.tlon,
            dims=["ny", "nx"],
            attrs={
                "long_name": "array of t-grid longitudes",
                "units": self._grid.tlon.attrs.get("units", "degrees_east"),
            },
        )

        ds["mask"] = xr.DataArray(
            self.tmask.astype(np.int32),
            dims=["ny", "nx"],
            attrs={
                "long_name": "landsea mask at t points: 1 ocean, 0 land",
                "units": "nondim",
            },
        )

        ds["depth"] = xr.DataArray(
            self._depth.data,
            dims=["ny", "nx"],
            attrs={"long_name": "t-grid cell depth", "units": "m"},
        )

        ds.to_netcdf(file_path)

    def write_cice_grid(self, file_path):
        """
        Write the CICE grid file in netcdf format. The written file is
        to be read in by CICE during runtime.

        Parameters
        ----------
        file_path: str
            Path to CICE grid file to be written.
        """

        assert (
            "degrees" in self._grid.tlat.attrs.get("units", "degrees_north") and "degrees" in self._grid.tlon.attrs.get("units", "degrees_east")
        ), "Unsupported coord"

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["title"] = "CICE grid file"

        ny = self._grid.ny
        nx = self._grid.nx

        ds["ulat"] = xr.DataArray(
            np.deg2rad(self._grid.qlat[1:, 1:].data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "U grid center latitude",
                "units": "radians",
                "bounds": "latu_bounds",
            },
        )

        ds["ulon"] = xr.DataArray(
            np.deg2rad(self._grid.qlon[1:, 1:].data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "U grid center longitude",
                "units": "radians",
                "bounds": "lonu_bounds",
            },
        )

        ds["tlat"] = xr.DataArray(
            np.deg2rad(self._grid.tlat.data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "T grid center latitude",
                "units": "degrees_north",
                "bounds": "latt_bounds",
            },
        )

        ds["tlon"] = xr.DataArray(
            np.deg2rad(self._grid.tlon.data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "T grid center longitude",
                "units": "degrees_east",
                "bounds": "lont_bounds",
            },
        )

        ds["htn"] = xr.DataArray(
            self._grid.dxCv.data * 100.0,
            dims=["nj", "ni"],
            attrs={
                "long_name": "T cell width on North side",
                "units": "cm",
                "coordinates": "TLON TLAT",
            },
        )

        ds["hte"] = xr.DataArray(
            self._grid.dyCu.data * 100,
            dims=["nj", "ni"],
            attrs={
                "long_name": "T cell width on East side",
                "units": "cm",
                "coordinates": "TLON TLAT",
            },
        )

        ds["angle"] = xr.DataArray(
            np.deg2rad(
                self._grid.angle_q.data[1:,1:] # Slice the q-grid from MOM6 (which is u-grid in CICE/POP) to CICE/POP convention, the top right of the t points
            ),
            dims=["nj", "ni"],
            attrs={
                "long_name": "angle grid makes with latitude line on U grid",
                "units": "radians",
                "coordinates": "ULON ULAT",
            },
        )

        ds["anglet"] = xr.DataArray(
            np.deg2rad(
                self._grid.angle.data
            ),
            dims=["nj", "ni"],
            attrs={
                "long_name": "angle grid makes with latitude line on T grid",
                "units": "radians",
                "coordinates": "TLON TLAT",
            },
        )

        ds["kmt"] = xr.DataArray(
            self.tmask.astype(np.float32),
            dims=["nj", "ni"],
            attrs={
                "long_name": "mask of T grid cells",
                "units": "unitless",
                "coordinates": "TLON TLAT",
            },
        )

        ds.to_netcdf(file_path)

    def write_scrip_grid(self, file_path, title=None):
        """
        Write the SCRIP grid file. In latest CESM versions, SCRIP grid files are
        no longer required and are replaced by ESMF mesh files. However, SCRIP
        files are still needed to generate custom ocean-runoff mapping files.

        Parameters
        ----------
        file_path: str
            Path to SCRIP file to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["Conventions"] = "SCRIP"
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title

        ds["grid_dims"] = xr.DataArray(
            np.array([self._grid.nx, self._grid.ny]).astype(np.int32),
            dims=["grid_rank"],
        )
        ds["grid_center_lat"] = xr.DataArray(
            self._grid.tlat.data.flatten(),
            dims=["grid_size"],
            attrs={"units": self._grid.supergrid.dict["axis_units"]},
        )
        ds["grid_center_lon"] = xr.DataArray(
            self._grid.tlon.data.flatten(),
            dims=["grid_size"],
            attrs={"units": self._grid.supergrid.dict["axis_units"]},
        )
        ds["grid_imask"] = xr.DataArray(
            self.tmask.data.astype(np.int32).flatten(),
            dims=["grid_size"],
            attrs={"units": "unitless"},
        )

        ds["grid_corner_lat"] = xr.DataArray(
            np.zeros((ds.sizes["grid_size"], 4)),
            dims=["grid_size", "grid_corners"],
            attrs={"units": self._grid.supergrid.dict["axis_units"]},
        )
        ds["grid_corner_lon"] = xr.DataArray(
            np.zeros((ds.sizes["grid_size"], 4)),
            dims=["grid_size", "grid_corners"],
            attrs={"units": self._grid.supergrid.dict["axis_units"]},
        )

        i_range = range(self._grid.nx)
        j_range = range(self._grid.ny)
        j, i = np.meshgrid(j_range, i_range, indexing="ij")
        k = j * self._grid.nx + i

        ds["grid_corner_lat"].data[k] = np.stack((
            self._grid.qlat.data[j, i],
            self._grid.qlat.data[j, i + 1],
            self._grid.qlat.data[j + 1, i + 1],
            self._grid.qlat.data[j + 1, i]
        ), axis=-1)

        ds["grid_corner_lon"].data[k] = np.stack((
            self._grid.qlon.data[j, i],
            self._grid.qlon.data[j, i + 1],
            self._grid.qlon.data[j + 1, i + 1],
            self._grid.qlon.data[j + 1, i]
        ), axis=-1)

        ds["grid_area"] = xr.DataArray(
            cell_area_rad(ds.grid_corner_lon.data, ds.grid_corner_lat.data),
            dims=["grid_size"],
            attrs={"units": "radians^2"},
        )

        ds.to_netcdf(file_path)

    def write_esmf_mesh(self, file_path, title=None):
        """
        Write the ESMF mesh file

        Parameters
        ----------
        file_path: str
            Path to ESMF mesh file to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["gridType"] = "unstructured mesh"
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title

        tlon_flat = self._grid.tlon.data.flatten()
        tlat_flat = self._grid.tlat.data.flatten()
        ncells = len(tlon_flat)  # i.e., elementCount in ESMF mesh nomenclature

        coord_units = self._grid.supergrid.dict["axis_units"]

        ds["centerCoords"] = xr.DataArray(
            [[tlon_flat[i], tlat_flat[i]] for i in range(ncells)],
            dims=["elementCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["numElementConn"] = xr.DataArray(
            np.full(ncells, 4).astype(np.int8),
            dims=["elementCount"],
            attrs={"long_name": "Node indices that define the element connectivity"},
        )
        
        ds["elementArea"] = xr.DataArray(
            self._grid.tarea.data.flatten(),
            dims=["elementCount"],
            attrs={"units": self._grid.tarea.attrs.get("units", "unknown")},
        )

        ds["elementMask"] = xr.DataArray(
            self.tmask.data.astype(np.int32).flatten(), dims=["elementCount"]
        )

        i0 = 1  # start index for node id's

        if self._grid.is_tripolar(self._grid._supergrid):
            
            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data[:, :-1].flatten()[:-(nx//2-1)]
            qlat_flat = self._grid.qlat.data[:, :-1].flatten()[:-(nx//2-1)]
            nnodes = len(qlon_flat)
            assert nnodes + (nx//2-1) == nx * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            def get_element_conn(i):
                is_final_column = (i+1) % nx == 0
                on_top_row = i // nx == ny-1
                on_second_half_of_stitch = on_top_row and (i%nx) >= nx//2

                # lower left corner
                ll = i0 + i % nx + (i // nx) * (nx)

                # lower right corner
                lr = ll + 1
                if is_final_column:
                    lr -= nx

                # upper right corner
                ur = lr+nx
                if on_second_half_of_stitch and not is_final_column:
                    ur -= 2 * (i % nx + 1 - nx // 2)

                # upper left corner
                ul = ll+nx
                if on_second_half_of_stitch:
                    ul = ur+1

                return [ll, lr, ur, ul]

        elif self._grid.supergrid.dict["cyclic_x"] == True:

            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data[:, :-1].flatten()
            qlat_flat = self._grid.qlat.data[:, :-1].flatten()
            nnodes = len(qlon_flat)
            assert nnodes == nx * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx),
                i0 + i % nx + (i // nx) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx),
            ]

        else: # non-cyclic grid

            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data.flatten()
            qlat_flat = self._grid.qlat.data.flatten()
            nnodes = len(qlon_flat)
            assert nnodes == (nx + 1) * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx + 1),
                i0 + i % nx + (i // nx) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1),
            ]


        ds["nodeCoords"] = xr.DataArray(
            np.column_stack((qlon_flat, qlat_flat)),
            dims=["nodeCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["elementConn"] = xr.DataArray(
            np.array([get_element_conn(i) for i in range(ncells)]).astype(np.int32),
            dims=["elementCount", "maxNodePElement"],
            attrs={
                "long_name": "Node indices that define the element connectivity",
                "start_index": np.int32(i0),
            },
        )

        self.mesh_path = file_path
        ds.to_netcdf(self.mesh_path)
