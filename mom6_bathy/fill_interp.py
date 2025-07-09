import numpy as np
import xarray as xr
import scipy
from mom6_bathy.aux import check_lon_range
from mom6_bathy.grid import Grid
import xesmf as xe
from os.path import isfile


def super_interp(src_lat, src_lon, data, spr_lat, spr_lon):
    """
    Bilinear Interpolation of data from src_lat, src_lon to spr_lat, spr_lon. Does not work for global grids
    """
    assert check_lon_range(src_lon) == check_lon_range(
        spr_lon
    ), "Longitude ranges must match"

    data = data.values

    src = {"lon": src_lon, "lat": src_lat}
    dst = {"lon": spr_lon.squeeze(), "lat": spr_lat.squeeze()}
    regridder = xe.Regridder(
        src,
        dst,
        "bilinear",
        filename="bilin_weights.nc",
        reuse_weights=isfile("bilin_weights.nc"),
    )
    result = regridder(data)
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
