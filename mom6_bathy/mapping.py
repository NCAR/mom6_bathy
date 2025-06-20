#!/usr/bin/env python3

import os
import argparse
import xesmf as xe
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

from mom6_bathy.aux import get_mesh_dimensions, cell_area_rad

def grid_from_esmf_mesh(mesh: xr.Dataset | str | Path) -> "Grid":
    """Given an ESMF mesh where the grid metrics are stored in 1D (flattened) arrays,
    compute the dimensions of the 2D grid and return a 2D horizontal grid dataset
    containing the longitude, latitude, and mask of the grid points which are all that
    is needed to create a regridder.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    ds : xr.Dataset
        A 2D horizontal grid dataset containing the longitude, latitude, and mask of the grid points.
    """

    if not isinstance(mesh, xr.Dataset):
        assert isinstance(mesh, (Path, str)) and Path(mesh).exists(), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    nx, ny = get_mesh_dimensions(mesh)

    lon = mesh['centerCoords'][:, 0].values.reshape((ny, nx))
    lat = mesh['centerCoords'][:, 1].values.reshape((ny, nx))
    mask = mesh['elementMask'].values.reshape((ny, nx))

    ds = xr.Dataset(
        {
            'lon': (('nlat', 'nlon'), lon),
            'lat': (('nlat', 'nlon'), lat),
            'mask': (('nlat', 'nlon'), mask),
        },
        coords={'nlat': np.arange(ny), 'nlon': np.arange(nx)}
    )

    return ds

def extract_coastline_mask(horiz_grid):
    """Given a 2D horizontal grid dataset, extract the coastline mask.

    Parameters
    ----------
    horiz_grid : xr.Dataset
        A 2D horizontal grid dataset containing the longitude, latitude, and mask of the grid points.
        This dataset may be obtained from the grid_from_esmf_mesh function by passing an ESMF mesh.

    Returns
    -------
    da_coastline_mask : xr.DataArray
        A 2D DataArray containing the coastline mask.
    """

    mask = horiz_grid['mask']

    # Apply padding to facilitate the diff operation
    mask_padded = np.pad(mask, pad_width=1, mode='wrap')

    coastline_mask = np.where(
        (
            (np.diff(mask_padded[:-1, 1:-1], axis=0) == 1) | (np.diff(mask_padded[1:, 1:-1], axis=0) == -1) |
            (np.diff(mask_padded[1:-1, :-1], axis=1) == 1) | (np.diff(mask_padded[1:-1, 1:], axis=1) == -1)
        ),
        1, 0
    )

    da_coastline_mask = mask.copy(data=coastline_mask)
    return da_coastline_mask


def sum_weights(regridder, stride=1):
    nx_in, ny_in = regridder.grid_in.get_shape()
    nx_out, ny_out = regridder.grid_out.get_shape()

    s = sum([(regridder.weights[:,i].data.reshape((ny_out,nx_out)).todense()) for i in range(0,ny_in*nx_in,stride)])
    da = xr.DataArray(s, dims=['nlat', 'nlon'], coords={'nlat': range(ny_out), 'nlon': range(nx_out)})
    return da


def _construct_vertex_coords(mesh):
    """Construct the vertex coordinates for a given mesh that's participating in a
    mapping generation. The mesh could be src or dst mesh.

    Parameters
    ----------
    mesh : xr.Dataset
        The ESMF mesh dataset.
    Returns
    -------
    xv_data : np.ndarray
        The vertex coordinates in the x-direction.
    yv_data : np.ndarray
        The vertex coordinates in the y-direction.
    """

    xv_data = np.full((len(mesh.centerCoords.data), 4), np.nan)
    yv_data = np.full((len(mesh.centerCoords.data), 4), np.nan)

    element_conn = mesh.elementConn.data - 1  # one-based to zero-based indexing
    node_coords = mesh.nodeCoords.data

    for e, nodes in enumerate(element_conn):
        if np.isnan(nodes[3]):
            valid_nodes = nodes[:3][~np.isnan(nodes[:3])].astype(int)
            xv_data[e, :3] = node_coords[valid_nodes, 0]
            xv_data[e, 3] = xv_data[e, 2]
            yv_data[e, :3] = node_coords[valid_nodes, 1]
            yv_data[e, 3] = yv_data[e, 2]
        else:
            valid_nodes = nodes.astype(int)
            xv_data[e, :] = node_coords[valid_nodes, 0]
            yv_data[e, :] = node_coords[valid_nodes, 1]
    
    return xv_data, yv_data

def generate_ESMF_map(src_mesh, dst_mesh, filename, weights=None, weights_esmpy=None, weights_coo=None, area_normalization=False):
    """Based on a given source mesh, destination mesh, and weights, generate an ESMF map file.
    
    Parameters
    ----------
    src_mesh : str, xr.Dataset
        ESMF mesh object or path to the source ESMF mesh file.
    dst_mesh : str, xr.Dataset
        ESMF mesh object or path to the destination ESMF mesh file.
    filename : str
        Path to the output ESMF map file.
    weights : xr.DataArray
        xESMF-generated weights to be used for the regridding.
    weights_esmpy:
        esmpy-generated weights to be used for the regridding.
    weights_coo : scipy.sparse.coo_matrix
        COO format sparse matrix containing the weights.
    regrid_method : str, optional
        Regridding method, by default 'bilinear'.
    area_normalization : bool, optional
        Whether to multiply the weights by (area_source / area_destination), by default False.

    Returns
    -------
    None
    """

    if isinstance(src_mesh, str):
        src_mesh = xr.open_dataset(src_mesh)
    if isinstance(dst_mesh, str):
        dst_mesh = xr.open_dataset(dst_mesh)
    
    if weights is weights_esmpy is weights_coo is None:
        raise ValueError("At least one of weights, weights_esmpy, or weights_coo must be provided.")
    if weights is not None:
        assert weights_esmpy is None, "Cannot provide both weights and weights_esmpy"
        assert weights_coo is None, "Cannot provide both weights and weights_coo"
        assert isinstance(weights, xr.DataArray), "weights must be an xarray DataArray"
    if weights_esmpy is not None:
        assert weights_coo is None, "Cannot provide both weights and weights_coo"
        assert isinstance(weights_esmpy, (xr.Dataset, str)), "weights_esmpy must be an xarray Dataset or a path to a file"
        if isinstance(weights_esmpy, str):
            weights_esmpy = xr.open_dataset(weights_esmpy)
        if not isinstance(weights_esmpy, coo_matrix):
            assert {'S', 'row', 'col'}.issubset(weights_esmpy), "weights_esmpy must contain 'S', 'row', and 'col'"
    if weights_coo is not None:
        assert isinstance(weights_coo, coo_matrix), "weights_coo must be a scipy sparse COO matrix"

    # From 1D ESMF mesh to 2D grid
    src_grid = grid_from_esmf_mesh(src_mesh)
    dst_grid = grid_from_esmf_mesh(dst_mesh)

    # 1/3: Source Domain Fields
    # -------------------

    xc_a = xr.DataArray(
        src_mesh.centerCoords.data[:, 0],
        dims=['n_a'],
        attrs={
            'long_name': 'longitude of grid cell center (input)',
            'units': 'degrees east'
        }
    )

    yc_a = xr.DataArray(
        src_mesh.centerCoords.data[:, 1],
        dims=['n_a'],
        attrs={
            'long_name': 'latitude of grid cell center (input)',
            'units': 'degrees north'
        }
    )

    xv_a_data, yv_a_data = _construct_vertex_coords(src_mesh)

    xv_a = xr.DataArray(
        xv_a_data,
        dims=['n_a', 'nv_a'],
        attrs={
            'long_name': 'longitude of grid cell verticies (input)',
            'units': 'degrees east'
        }
    )

    yv_a = xr.DataArray(
        yv_a_data,
        dims=['n_a', 'nv_a'],
        attrs={
            'long_name': 'latitude of grid cell verticies (input)',
            'units': 'degrees north'
        }
    )

    mask_a = xr.DataArray(
        src_mesh.elementMask.data,
        dims=['n_a'],
        attrs={
            'long_name': 'domain mask (input)',
        }
    )

    area_a = xr.DataArray(
        cell_area_rad(xv_a, yv_a),
        dims=['n_a'],
        attrs={
            'long_name': 'area of cell (input)',
            'units': 'radians^2'
        }
    )

    frac_a = xr.DataArray(
        src_mesh.elementMask.data.astype(np.float64),
        dims=['n_a'],
        attrs={
            'long_name': 'fraction of domain intersection (input)',
            #'units': ''
        }
    )

    src_grid_dims = xr.DataArray(
        np.array(src_grid.mask.shape[::-1]).astype(np.int32),
        dims=['src_grid_rank'],
        #attrs={
        #    'long_name': 'dimensions of the source grid',
        #}
    )

    nj_a = xr.DataArray(
        [i+1 for i in range(src_grid.mask.shape[0])],
        dims=['nj_a'],
    )

    ni_a = xr.DataArray(
        [i+1 for i in range(src_grid.mask.shape[1])],
        dims=['ni_a'],
    )

    # 2/3: Destination Domain Fields
    # -------------------

    xc_b = xr.DataArray(
        dst_mesh.centerCoords.data[:, 0],
        dims=['n_b'],
        attrs={
            'long_name': 'longitude of grid cell center (output)',
            'units': 'degrees east'
        }
    )

    yc_b = xr.DataArray(
        dst_mesh.centerCoords.data[:, 1],
        dims=['n_b'],
        attrs={
            'long_name': 'latitude of grid cell center (output)',
            'units': 'degrees north'
        }
    )

    xv_b_data, yv_b_data = _construct_vertex_coords(dst_mesh)

    xv_b = xr.DataArray(
        xv_b_data,
        dims=['n_b', 'nv_b'],
        attrs={
            'long_name': 'longitude of grid cell verticies (output)',
            'units': 'degrees east'
        }
    )

    yv_b = xr.DataArray(
        yv_b_data,
        dims=['n_b', 'nv_b'],
        attrs={
            'long_name': 'latitude of grid cell verticies (output)',
            'units': 'degrees north'
        }
    )

    mask_b = xr.DataArray(
        dst_mesh.elementMask.data,
        dims=['n_b'],
        attrs={
            'long_name': 'domain mask (output)',
        }
    )

    area_b = xr.DataArray(
        cell_area_rad(xv_b, yv_b),
        dims=['n_b'],
        attrs={
            'long_name': 'area of cell (output)',
            'units': 'radians^2'
        }
    )

    frac_b = xr.DataArray(
        dst_mesh.elementMask.data.astype(np.float64),
        dims=['n_b'],
        attrs={
            'long_name': 'fraction of domain intersection (output)',
            #'units': ''
        }
    )

    dst_grid_dims = xr.DataArray(
        np.array(dst_grid.mask.shape[::-1]).astype(np.int32),
        dims=['dst_grid_rank'],
        #dst_grid_dimsattrs={
        #    'long_name': 'dimensions of the destination grid',
        #}
    )

    nj_b = xr.DataArray(
        [i+1 for i in range(dst_grid.mask.shape[0])],
        dims=['nj_b'],
    )

    ni_b = xr.DataArray(
        [i+1 for i in range(dst_grid.mask.shape[1])],
        dims=['ni_b'],
    )

    # 3/3: Weights
    # -------------------

    if weights is not None: # xESMF weights
        w = weights.data.copy()
        col_data = w.coords[1, :] + 1
        row_data = w.coords[0, :] + 1
    elif weights_esmpy is not None: # esmpy weights
        w = weights_esmpy.S.copy()
        col_data = weights_esmpy.col.data
        row_data = weights_esmpy.row.data
    elif weights_coo is not None: # scipy sparse COO matrix
        w = weights_coo.data
        col_data = weights_coo.col + 1
        row_data = weights_coo.row + 1

    if area_normalization:
        w.data *= area_a.data[col_data - 1] / area_b.data[row_data - 1]

    S = xr.DataArray(
        w.data,
        dims=['n_s'],
        attrs={
            'long_name': 'sparse matrix for mapping S:a->b',
        }
    )

    col = xr.DataArray(
        col_data,
        dims=['n_s'],
        attrs={
            'long_name': 'column corresponding to matrix elements',
        }
    )

    row = xr.DataArray(
        row_data,
        dims=['n_s'],
        attrs={
            'long_name': 'row corresponding to matrix elements',
        }
    )

    # Drop NaN values from S, col, and row
    non_nan_indices = np.where(~np.isnan(S))[0]
    S_new = S[non_nan_indices]
    row_new = row[non_nan_indices]
    col_new = col[non_nan_indices]

    # Create the dataset and write to a netCDF file
    # ------------------------------------------------

    ds = xr.Dataset(
        {
            'xc_a': xc_a,
            'yc_a': yc_a,
            'xv_a': xv_a,
            'yv_a': yv_a,
            'mask_a': mask_a,
            'area_a': area_a,
            'frac_a': frac_a,
            'src_grid_dims': src_grid_dims,
            'nj_a': nj_a,
            'ni_a': ni_a,
            'xc_b': xc_b,
            'yc_b': yc_b,
            'xv_b': xv_b,
            'yv_b': yv_b,
            'mask_b': mask_b,
            'area_b': area_b,
            'frac_b': frac_b,
            'dst_grid_dims': dst_grid_dims,
            'nj_b': nj_b,
            'ni_b': ni_b,
            'S': S_new,
            'col': col_new,
            'row': row_new,
        }
    )

    ds.attrs['title'] = 'mom6_bathy mapping generation'
    #ds.attrs['normalization'] = 'conservative' # todo
    #ds.attrs['map_method'] = 'nearest neighbor smoothing'
    ds.attrs['history'] = 'generated by mom6_bathy'
    ds.attrs['conventions'] = 'NCAR-CCSM'
    #ds.attrs['domain_a'] = '[PATH_TO_SRC_MESH]'
    #ds.attrs['domain_b'] = '[PATH_TO_DST_MESH]'

    ds.to_netcdf(
        filename,
        format='NETCDF3_64BIT',
        encoding={var: {'_FillValue': None} for var in ds.data_vars}
    )



def generate_ESMF_map_via_esmpy(src_mesh_path, dst_mesh_path, mapping_file, method, area_normalization):
    """Generate an ESMF mapping file using esmpy.
    
    Parameters
    ----------
    src_mesh_path : str or Path
        Path to the source ESMF mesh file.
    dst_mesh_path : str or Path
        Path to the destination ESMF mesh file.
    mapping_file : str or Path
        Path to the output ESMF mapping file to be created.
    method : str
        Regridding method to use. Options are 'nearest_d2s', 'nearest_s2d', 'bilinear', 'conservative'.
    area_normalization : bool
        Whether to apply area normalization to the weights.
    """

    import esmpy

    assert isinstance(src_mesh_path, (str, Path)), "src_mesh_path must be a path to an existing file"
    assert isinstance(dst_mesh_path, (str, Path)), "dst_mesh_path must be a path to an existing file"

    match method:
        case 'nearest_d2s':
            method = esmpy.RegridMethod.NEAREST_DTOS
        case 'nearest_s2d':
            method = esmpy.RegridMethod.NEAREST_STOD
        case 'bilinear':
            raise NotImplementedError("Bilinear regridding is not yet tested.")
        case 'conservative':
            raise NotImplementedError("Conservative regridding is not yet tested.")
        case _:
            raise ValueError(f"Invalid regridding method: {method}")

    # Create src and dst meshes and fields
    src_mesh = esmpy.Mesh(
        filename=src_mesh_path,
        filetype=esmpy.FileFormat.ESMFMESH,
        mask_flag = esmpy.api.constants.MeshLoc.ELEMENT,
        varname = "elementMask"
    )

    src_field = esmpy.Field(src_mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    dst_mesh = esmpy.Mesh(
        filename=dst_mesh_path,
        filetype=esmpy.FileFormat.ESMFMESH,
        mask_flag = esmpy.api.constants.MeshLoc.ELEMENT,
        varname = "elementMask"
    )

    dst_field = esmpy.Field(dst_mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    # Apply mask:
    maskval = 0.0  # Value to use for masked elements
    src_mesh_ds = xr.open_dataset(src_mesh_path)
    src_field.data[:] = np.where(src_mesh_ds.elementMask.data == 0, maskval, 1.0)
    dst_mesh_ds = xr.open_dataset(dst_mesh_path)
    dst_field.data[:] = np.where(dst_mesh_ds.elementMask.data == 0, maskval, 1.0)

    # Run the regridder
    if os.path.exists(mapping_file):
        os.remove(mapping_file)

    # Create regridder and save the weights to the mapping file temporarily
    # This file will later be extended to include the full ESMF map fields
    esmpy.Regrid(
        src_field,
        dst_field,
        filename=mapping_file,
        src_mask_values=np.array([maskval]),
        dst_mask_values=np.array([maskval]),
        regrid_method=method,
        unmapped_action=esmpy.UnmappedAction.ERROR,
    )

    # Now, read in the weights from the mapping file
    weights_ds = xr.open_dataset(mapping_file)

    generate_ESMF_map(
        src_mesh=src_mesh_path,
        dst_mesh=dst_mesh_path,
        filename=mapping_file,
        weights_esmpy=weights_ds,
        area_normalization=area_normalization
    )


def compute_smoothing_weights(mesh_ds, rmax, fold=1.0, xv_data=None, yv_data=None):
    """Compute smoothing weights for a given mesh dataset, using a radius in kilometers.

    Parameters
    ----------
    mesh_ds : xr.Dataset
        The ESMF mesh dataset.
    rmax : float
        Maximum distance for smoothing weights, in kilometers.
    fold : float
        Fold factor (km) determining the strength of smoothing based on distance.
    xv_data : np.ndarray, optional
        The x-coordinates of the vertices. If not provided, they will be computed from the mesh dataset.
    yv_data : np.ndarray, optional
        The y-coordinates of the vertices. If not provided, they will be computed from the mesh dataset.

    Returns
    -------
    weights : scipy.sparse.coo_matrix
        The computed smoothing weights in coordinate (COO) format.
    """

    if not isinstance(mesh_ds, xr.Dataset):
        raise ValueError("mesh_ds must be an xarray Dataset.")

    # Extract coordinates and mask
    coords = mesh_ds['centerCoords'].values
    if xv_data is None or yv_data is None:
        xv_data, yv_data = _construct_vertex_coords(mesh_ds)
    areas = cell_area_rad(xv_data, yv_data)
    mask = mesh_ds['elementMask'].values
    mask_bool = mask != 0

    # Convert lon/lat (degrees) to radians
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])

    # Earth's radius in kilometers
    R_earth = 6371.0

    # Convert lon/lat to 3D Cartesian coordinates for great-circle distance calculation
    x = R_earth * np.cos(lat) * np.cos(lon)
    y = R_earth * np.cos(lat) * np.sin(lon)
    z = R_earth * np.sin(lat)
    xyz = np.stack([x, y, z], axis=1)

    # Build KDTree in 3D Cartesian space
    tree = cKDTree(xyz)
    indices = tree.query_ball_tree(tree, rmax)

    row_indices, col_indices, data = [], [], []

    for i, neighbors in enumerate(indices):
        if not mask_bool[i]:
            continue
        neighbors = np.array(neighbors)
        neighbors = neighbors[mask_bool[neighbors]]
        row_indices.extend([i] * len(neighbors))
        col_indices.extend(neighbors)
        # Compute great-circle distances in km
        d_xyz = xyz[i] - xyz[neighbors]
        # Chord length
        chord = np.linalg.norm(d_xyz, axis=1)
        # Convert chord length to arc length (distance on sphere)
        arc = 2 * R_earth * np.arcsin(np.clip(chord / (2 * R_earth), 0, 1))
        weights_data = np.exp(-arc / fold)
        # Normalize by area
        weights_data /= np.sum(weights_data * (areas[neighbors] / areas[i]))
        data.extend(weights_data)

    weights = coo_matrix((data, (row_indices, col_indices)), shape=(len(coords), len(coords)))
    return weights


def main(args):

    if not isinstance(args.src_mesh, (str, Path)) or not Path(args.src_mesh).exists():
        raise ValueError("src_mesh must be a path to an existing ESMF mesh file.")
    if not isinstance(args.dst_mesh, (str, Path)) or not Path(args.dst_mesh).exists():
        raise ValueError("dst_mesh must be a path to an existing ESMF mesh file.")
    if not isinstance(args.mapping_file, (str, Path)) or Path(args.mapping_file).exists():
        raise ValueError("mapping_file must be a path to a new ESMF mapping file.")
    if args.use_esmpy:

        generate_ESMF_map_via_esmpy(
            src_mesh_path=args.src_mesh,
            dst_mesh_path=args.dst_mesh,
            mapping_file=args.mapping_file,
            method=args.method,
            area_normalization=args.area_normalization
        )

    else:
        raise NotImplementedError("xESMF-based mapping generation is not yet implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map 1D ESMF mesh to a 2D horizontal grid")
    parser.add_argument("--src_mesh", type=str, help="Path to the source ESMF mesh file or dataset")
    parser.add_argument("--dst_mesh", type=str, help="Path to the destination ESMF mesh file or dataset")
    parser.add_argument("--mapping_file", type=str, help="Path to the output ESMF mapping file to be created")
    parser.add_argument("--method", type=str, choices=['nearest_d2s', 'nearest_s2d', 'bilinear', 'conservative'],
                        help="Regridding method to use", default='nearest_d2s')
    parser.add_argument("--area_normalization", action="store_true", 
                        help="Whether to apply area normalization to the weights", default=False)
    parser.add_argument("--use_esmpy", action="store_true", 
                        help="Whether to use esmpy for regridding instead of xESMF", default=False)

    args = parser.parse_args()
    main(args)