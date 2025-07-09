import numpy as np
import xarray as xr
import scipy
from mom6_bathy.aux import check_lon_range
from mom6_bathy.grid import Grid
import xesmf as xe
from os.path import isfile


def latlon2ji(src_lat, src_lon, lat, lon):
    nj, ni = len(src_lat), len(src_lon)
    src_x0 = int((src_lon[0] + src_lon[-1]) / 2 + 0.5) - 180.0
    j = np.maximum(0, np.floor(((lat + 90.0) / 180.0) * nj - 0.5).astype(int))
    i = np.mod(np.floor(((lon - src_x0) / 360.0) * ni - 0.5), ni).astype(int)
    jp1 = np.minimum(nj - 1, j + 1)
    ip1 = np.mod(i + 1, ni)
    return j, i, jp1, ip1


def latlon2ji_simple(src_lat, src_lon, lat, lon):
    """
    Assumes the lon and late are in the correct range (no negative lats, same lon format between src_lon  & spr_lon)
    """
    lat0, lat1 = src_lat[0], src_lat[-1]
    lon0, lon1 = src_lon[0], src_lon[-1]
    # Check ranges
    if np.any(lat < lat0) or np.any(lat > lat1):
        raise ValueError(
            f"Latitude values out of bounds: must be between {lat0} and {lat1}"
        )

    if np.any(lon < lon0) or np.any(lon > lon1):
        raise ValueError(
            f"Longitude values out of bounds: must be between {lon0} and {lon1}"
        )

    nj, ni = len(src_lat), len(src_lon)

    lat_range = src_lat[-1] - src_lat[0]
    lon_range = src_lon[-1] - src_lon[0]

    j = np.floor(((lat - src_lat[0]) / lat_range) * nj - 0.5).astype(int)
    i = np.floor(((lon - src_lon[0]) / lon_range) * ni - 0.5).astype(int)

    jp1 = j + 1
    ip1 = i + 1

    return j, i, jp1, ip1


def super_interp(src_lat, src_lon, data, spr_lat, spr_lon):
    """
    Bilinear Interpolation of data from src_lat, src_lon to spr_lat, spr_lon. Does not assume periodicity.
    """
    assert check_lon_range(src_lon) == check_lon_range(
        spr_lon
    ), "Longitude ranges must match"

    data = data.values
    dy, dx = np.mean(np.diff(src_lon)), np.mean(np.diff(src_lat))

    j0, i0, j1, i1 = latlon2ji_simple(src_lat, src_lon, spr_lat, spr_lon)

    # Create dummy grid for chlorophyll source data for access to get_indices
    chl_source_grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=1.0,
        ystart=7.0,
        leny=1.0,
        name="seawifs",
    )

    # Set Correct CHL coordinates
    lon2d, lat2d = np.meshgrid(src_lon, src_lat)  # shape (nj, ni)

    chl_source_grid.tlon = xr.DataArray(lon2d)
    chl_source_grid.tlat = xr.DataArray(lat2d)

    # Reset KDTree
    chl_source_grid._kdtree = None

    # # Get indexes for source data
    # j0,i0 = chl_source_grid.get_indices(spr_lat.ravel(),spr_lon.ravel())
    # j0 = j0.reshape(spr_lat.shape)
    # i0 = i0.reshape(spr_lon.shape)
    # j1 = j0 + 1
    # i1 = i0 + 1

    src = {"lon": lon2d, "lat": lat2d}
    dst = {"lon": spr_lon.squeeze(), "lat": spr_lat.squeeze()}
    regridder = xe.Regridder(
        src,
        dst,
        "bilinear",
        filename="bilin_weights.nc",
        reuse_weights=isfile("bilin_weights.nc"),
    )
    result = regridder(data)

    def ydist(lat0, lat1):
        return np.abs(lat1 - lat0)

    def xdist(lon0, lon1):
        return np.abs(lon1 - lon0)

    w_e = xdist(src_lon[i0], spr_lon) / dx
    w_w = 1.0 - w_e
    w_n = ydist(src_lat[j0], spr_lat) / dy
    w_s = 1.0 - w_n

    r2 = (w_s * w_w * data[j0, i0] + w_n * w_e * data[j1, i1]) + (
        w_n * w_w * data[j1, i0] + w_s * w_e * data[j0, i1]
    )
    return result[..., np.newaxis, np.newaxis]


def fill_missing_data(idata, mask, maxiter=0, stabilizer=1.0e-14, tripole=False):
    """
    Returns data with masked values objectively interpolated except where mask==0. Does not work for periodic grids.

    Arguments:
    data - np.ma.array with mask==True where there is missing data or land.
    mask - np.array of 0 or 1, 0 for land, 1 for ocean.

    Returns a np.ma.array.
    """
    nj, ni = idata.shape
    fdata = np.nan_to_num(idata, nan=0.0)
    missing_j, missing_i = np.where(np.isnan(idata) & (mask > 0))

    n_missing = missing_i.size

    # ind contains column of matrix/row of vector corresponding to point [j,i]
    ind = np.zeros(fdata.shape, dtype=int) - int(1e6)
    ind[missing_j, missing_i] = np.arange(n_missing)
    A = scipy.sparse.lil_matrix((n_missing, n_missing))
    b = np.zeros((n_missing))
    ld = np.zeros((n_missing))
    A[range(n_missing), range(n_missing)] = 0.0
    for n in range(n_missing):
        j, i = missing_j[n], missing_i[n]
        im1 = max(i - 1, 0)
        ip1 = min(i + 1, ni - 1)
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, nj - 1)
        if j > 0 and mask[jm1, i] > 0:
            ld[n] -= 1.0
            ij = ind[jm1, i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[jm1, i]
        if i > 0 and mask[j, im1] > 0:
            ld[n] -= 1.0
            ij = ind[j, im1]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, im1]
        if i < ni - 1 and mask[j, ip1] > 0:
            ld[n] -= 1.0
            ij = ind[j, ip1]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, ip1]
        if j < nj - 1 and mask[jp1, i] > 0:
            ld[n] -= 1.0
            ij = ind[jp1, i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[jp1, i]
        if j == nj - 1 and mask[j, ni - 1 - i] > 0 and tripole:  # Tri-polar fold
            ld[n] -= 1.0
            ij = ind[j, ni - 1 - i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, ni - 1 - i]
    # Set leading diagonal
    b[ld >= 0] = 0.0
    A[range(n_missing), range(n_missing)] = ld - stabilizer

    A = scipy.sparse.csr_matrix(A)
    new_data = np.ma.array(fdata, mask=(mask == 0))
    if maxiter is None:
        x, info = scipy.sparse.linalg.bicg(A, b)
    elif maxiter == 0:
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        x, info = scipy.sparse.linalg.bicg(A, b, maxiter=maxiter)
    new_data[missing_j, missing_i] = x
    return new_data
