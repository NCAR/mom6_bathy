import numpy as np
import xarray as xr
from numba import jit
import matplotlib.pyplot as plt
import scipy


def lateral_fill(
    da_in,
    isvalid_mask,
    ltripole=False,
    tol=1.0e-4,
    use_sor=False,
    rc=1.8,
    max_iter=10000,
):
    """Perform lateral fill on xarray.DataArray

    Parameters
    ----------
    da_in : xarray.DataArray
      DataArray on which to fill NaNs. Fill is performed on the two
      rightmost dimenions. Grid is assumed periodic in `x` direction
      (last dimension).

    isvalid_mask : xarray.DataArray, boolean
      Valid values mask: `True` where data should be filled. Must have the
      same rightmost dimenions as `da_in`.

    ltripole : boolean, optional [default=False]
      Logical flag; if `True` then treat the top row of the grid as periodic
      in the sense of a tripole grid.

     tol : float, optional [default=1.0e-4]
      Convergence criteria: stop filling when values change is less or equal
      to `tol * var`; i.e. `delta <= tol * np.abs(var[j, i])`.

    use_sor: boolean, optional [default=False]
      switch to select SOR fill algorithm over progressive fill algorithm

    rc : float, optional [default=1.8, valid bounds=(1.0,2.0)]
       over-relaxation coefficient to use in SOR fill algorithm.
       Larger arrrays (or extent of region to be filled if not global)
       typically converge faster with larger coefficients.
       For completely land-filling a 1 deg. grid (360x180) a coefficient in
       the range 1.85-1.9 is near optimal.

    max_iter : integer, optional, [default=10000]
       maximum number of iterations to do before giving up if tol is not reached.

    Returns
    -------
    da_out : xarray.DataArray
      DataArray with NaNs filled by iterative smoothing.

    """
    dims_in = da_in.dims
    non_lateral_dims = dims_in[:-2]

    attrs = da_in.attrs
    encoding = da_in.encoding
    coords = da_in.coords
    
    da_in, isvalid_mask = xr.broadcast(da_in, isvalid_mask)

    if len(non_lateral_dims) > 0:
        da_in_stack = da_in.stack(non_lateral_dims=non_lateral_dims)
        da_out_stack = xr.full_like(da_in_stack, fill_value=np.nan)
        isvalid_mask_stack = isvalid_mask.stack(non_lateral_dims=non_lateral_dims)
        for i in range(da_in_stack.shape[-1]):
            arr = da_in_stack.data[:, :, i]
            da_out_stack[:, :, i] = lateral_fill_np_array(
                arr,
                isvalid_mask_stack.data[:, :, i],
                ltripole,
                tol,
                use_sor,
                rc,
                max_iter,
            )

        da_out = da_out_stack.unstack("non_lateral_dims").transpose(*dims_in)

    else:
        da_out = xr.full_like(da_in, fill_value=np.nan)
        da_out[:, :] = lateral_fill_np_array(
            da_in.data, isvalid_mask.values.astype(bool), ltripole, tol, use_sor, rc, max_iter
        )

    da_out.attrs = attrs
    da_out.encoding = encoding
    for k, da in coords.items():
        da_out[k].attrs = da.attrs

    return da_out


def lateral_fill_np_array(
    var, isvalid_mask, ltripole=False, tol=1.0e-4, use_sor=False, rc=1.8, max_iter=10000
):
    """Perform lateral fill on numpy.array

    Parameters [NB defaults set redundantly with lateral_fill above to allow this
                function to be called directly for numpy arrays]
    ----------

    var : numpy.array
      Array on which to fill NaNs. Fill is performed on the two
      rightmost dimenions. Grid is assumed periodic in `x` direction
      (last dimension). Only NaNs where isvalid_mask is True will be filled.

    isvalid_mask : numpy.array, boolean
      Valid values mask: `True` where data should be filled. Must have the
      same rightmost dimenions as `da_in`.

    ltripole : boolean, optional [default=False set in lateral_fill]
      Logical flag; if `True` then treat the top row of the grid as periodic
      in the sense of a tripole grid.

    tol : float, optional [default=1.0e-4 set in lateral_fill]
      Convergence criteria: stop filling when values change is less or equal
      to `tol * var`; i.e. `delta <= tol * np.abs(var[j, i])`.

    use_sor: boolean, optional [default=False set in lateral_fill]
      switch to select SOR fill algorithm

    rc : float, optional [default=1.8, valid bounds=(1.0,2.0)]
       over-relaxation coefficient to use in SOR fill algorithm.
       Larger arrrays (or extent of region to be filled if not global)
       typically converge faster with larger coefficients.
       For completely land-filling a 1 deg. grid (360x180) a coefficient in
       the range 1.85-1.9 is near optimal.

    max_iter : integer, optional, [default=10000 set in lateral_fill]
       maximum number of iterations to do before giving up if tol is not reached.

    Returns
    -------

    da_out : xarray.DataArray
      DataArray with NaNs filled by iterative smoothing.

    """

    assert (
        var.ndim == 2
    ), f"lateral_fill_np_array expects 2D array, but got {var.ndim}: {var.shape}"

    nlat, nlon = var.shape[-2:]

    var = var.copy()

    if use_sor:
        fillmask = np.isnan(var)  # Fill all NaNs
        keepNaNs = ~isvalid_mask & np.isnan(var)
        _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, ltripole)
        var[keepNaNs] = np.nan  # Replace NaNs in areas not designated for filling
    else:
        fillmask = np.isnan(var) & isvalid_mask
        _iterative_fill_POP_core(nlat, nlon, var, fillmask, tol, ltripole, max_iter)

    return var


@jit(nopython=True, parallel=True)
def _iterative_fill_POP_core(nlat, nlon, var, fillmask, tol, ltripole, max_iter):
    """Iterative smoothing algorithm."""

    done = False
    iter_cnt = 0

    work = np.empty_like(var)

    while not done:
        done = True
        iter_cnt += 1

        # assume bottom row is land, so skip it
        for j in range(1, nlat):
            jm1 = j - 1
            jp1 = j + 1

            for i in range(0, nlon):
                # assume periodic in x
                im1 = (i - 1) % nlon
                ip1 = (i + 1) % nlon

                work[j, i] = var[j, i]

                if not fillmask[j, i]:
                    continue

                numer = 0.0
                denom = 0.0

                # East
                if np.isfinite(var[j, ip1]):
                    numer += var[j, ip1]
                    denom += 1.0

                # North
                if j < nlat - 1:
                    if np.isfinite(var[jp1, i]):
                        numer += var[jp1, i]
                        denom += 1.0

                else:
                    # assume only tripole has non-land top row
                    if ltripole:
                        if np.isfinite(var[j, nlon - 1 - i]):
                            numer += var[j, nlon - 1 - i]
                            denom += 1.0

                # West
                if np.isfinite(var[j, im1]):
                    numer += var[j, im1]
                    denom += 1.0

                # South
                if np.isfinite(var[jm1, i]):
                    numer += var[jm1, i]
                    denom += 1.0

                # self
                if np.isfinite(var[j, i]):
                    numer += denom * var[j, i]
                    denom *= 2.0

                if denom > 0.0:
                    work[j, i] = numer / denom
                    if np.isnan(var[j, i]):
                        done = False
                    else:
                        delta = np.fabs(var[j, i] - work[j, i])
                        if delta > tol * np.abs(var[j, i]):
                            done = False

        var[1:nlat, :] = work[1:nlat, :]
        if iter_cnt > max_iter:
            done = True


@jit(nopython=True, parallel=True)
def _iterative_fill_sor(nlat, nlon, var, fillmask, tol, rc, max_iter, ltripole):
    """Iterative land fill algorithm via SOR solution of Laplace Equation."""

    # Compute a zonal mean to use as a first guess
    # Apprarently jit doesn't like masked arrays so loop it out

    zoncnt = np.zeros(nlat)
    zonavg = np.zeros(nlat)
    for j in range(0, nlat):
        zoncnt[j] = np.sum(np.where(fillmask[j, :], 0, 1))
        zonavg[j] = np.sum(np.where(fillmask[j, :], 0, var[j, :]))
        if zoncnt[j] != 0:
            zonavg[j] = zonavg[j] / zoncnt[j]

    # Fill missing zonal averages for rows that are entirely land
    for j in range(0, nlat - 1):  # northward pass
        if zoncnt[j] > 0 and zoncnt[j + 1] == 0:
            zoncnt[j + 1] = 1
            zonavg[j + 1] = zonavg[j]
    for j in range(nlat - 1, 0, -1):  # southward pass
        if zoncnt[j] > 0 and zoncnt[j - 1] == 0:
            zoncnt[j - 1] = 1
            zonavg[j - 1] = zonavg[j]

    # Replace the input array missing values with zonal average as first guess
    for j in range(0, nlat):
        for i in range(0, nlon):
            if fillmask[j, i]:
                var[j, i] = zonavg[j]

    # Now do the iterative 2D fill
    res = np.zeros((nlat, nlon))  # work array hold residuals
    res_max = tol
    iter_cnt = 0
    while iter_cnt < max_iter and res_max >= tol:
        res[:] = 0.0  # reset the residual to zero for this iteration

        # assume bottom row is all land, leave it set to zonal average
        # deal with top row separately below
        for j in range(1, nlat - 1):
            jm1 = j - 1
            jp1 = j + 1

            for i in range(0, nlon):
                if fillmask[j, i]:
                    im1 = (i - 1) % nlon  # assume periodic in x
                    ip1 = (i + 1) % nlon

                    # this is SOR
                    res[j, i] = (
                        var[j, ip1]
                        + var[j, im1]
                        + var[jm1, i]
                        + var[jp1, i]
                        - 4.0 * var[j, i]
                    )
                    var[j, i] = var[j, i] + rc * 0.25 * res[j, i]

        # do the top row if there was some valid data there in the input
        # otherwise leave it set to zonal average of northernmost row with valid data
        if zoncnt[nlat - 1] > 1:
            j = nlat - 1
            jm1 = j - 1
            jp1 = j
            for i in range(0, nlon):
                if fillmask[j, i]:
                    im1 = (i - 1) % nlon
                    ip1 = (i + 1) % nlon
                    io = nlon - 1 - i  # tripole b.c

                    if ltripole:  # use cross-pole periodicity
                        res[j, i] = (
                            var[j, ip1]
                            + var[j, im1]
                            + var[jp1, io]
                            + var[jm1, i]
                            - 4.0 * var[j, i]
                        )
                        var[j, i] = var[j, i] + rc * 0.25 * res[j, i]
                    else:  # do a 1D smooth on pole row
                        res[j, i] = var[j, ip1] + var[j, im1] - 2.0 * var[j, i]
                        var[j, i] = var[j, i] + rc * 0.5 * res[j, i]

        res_max = np.max(np.fabs(res)) / np.max(np.fabs(var))
        iter_cnt += 1


def super_sample_grid(ocn_qlat, ocn_qlon, ocn_mask, src_nj, src_ni):
    """
    Super-sample an ocean grid using bilinear interpolation.

    This function refines supergrid coordinates (ocn_qlat, ocn_qlon)
    to a finer resolution by subdividing each original grid cell into a higher-resolution
    `fac x fac` subgrid, where `fac` is automatically determined to match or exceed
    the target dimensions (`src_nj`, `src_ni`).

    Parameters
    ----------
    ocn_qlat : ndarray of shape (nj+1, ni+1)
        Latitude at the corners of each grid cell.
    ocn_qlon : ndarray of shape (nj+1, ni+1)
        Longitude at the corners of each grid cell.
    ocn_mask : ndarray of shape (nj, ni)
        Mask array indicating valid ocean cells (e.g., 1 for ocean, 0 for land).
    src_nj : int
        Desired number of rows in the upsampled grid.
    src_ni : int
        Desired number of columns in the upsampled grid.

    Returns
    -------
    lat : ndarray of shape (nj, fac, ni, fac)
        Refined latitude field with `fac x fac` points per original cell.
    lon : ndarray of shape (nj, fac, ni, fac)
        Refined longitude field with `fac x fac` points per original cell.

    Notes
    -----
    The output grids can be reshaped to `(nj * fac, ni * fac)` for use in plotting
    or remapping. The interpolation uses bilinear weights based on relative position
    within each original grid cell.
    """
    nj, ni = ocn_mask.shape
    fac = 1
    while fac * nj < src_nj and fac * ni < src_ni:
        fac += 1
    lon = np.zeros((nj, fac, ni, fac))
    lat = np.zeros((nj, fac, ni, fac))
    for j in range(fac):
        ya = (2 * j + 1) / (2 * fac)
        yb = 1.0 - ya
        for i in range(fac):
            xa = (2 * i + 1) / (2 * fac)
            xb = 1.0 - xa
            lon[:, j, :, i] = yb * (
                xb * ocn_qlon[:-1, :-1] + xa * ocn_qlon[:-1, 1:]
            ) + ya * (xb * ocn_qlon[1:, :-1] + xa * ocn_qlon[1:, 1:])
            lat[:, j, :, i] = yb * (
                xb * ocn_qlat[:-1, :-1] + xa * ocn_qlat[:-1, 1:]
            ) + ya * (xb * ocn_qlat[1:, :-1] + xa * ocn_qlat[1:, 1:])
    return lat, lon


def latlon2ji(src_lat, src_lon, lat, lon):
    nj, ni = len(src_lat), len(src_lon)
    src_x0 = int((src_lon[0] + src_lon[-1]) / 2 + 0.5) - 180.0
    j = np.maximum(0, np.floor(((lat + 90.0) / 180.0) * nj - 0.5).astype(int))
    i = np.mod(np.floor(((lon - src_x0) / 360.0) * ni - 0.5), ni).astype(int)
    jp1 = np.minimum(nj - 1, j + 1)
    ip1 = np.mod(i + 1, ni)
    return j, i, jp1, ip1


def super_interp(src_lat, src_lon, data, spr_lat, spr_lon):
    data = data.values
    nj, ni = data.shape
    dy, dx = 180.0 / nj, 360.0 / ni
    j0, i0, j1, i1 = latlon2ji(src_lat, src_lon, spr_lat, spr_lon)

    def ydist(lat0, lat1):
        return np.abs(lat1 - lat0)

    def xdist(lon0, lon1):
        return np.abs(np.mod((lon1 - lon0) + 180, 360) - 180)

    w_e = xdist(src_lon[i0], spr_lon) / dx
    w_w = 1.0 - w_e
    w_n = ydist(src_lat[j0], spr_lat) / dy
    w_s = 1.0 - w_n
    return (w_s * w_w * data[j0, i0] + w_n * w_e * data[j1, i1]) + (
        w_n * w_w * data[j1, i0] + w_s * w_e * data[j0, i1]
    )


def fill_missing_data(
    idata, mask, verbose=True, maxiter=0, debug=False, stabilizer=1.0e-14
):
    """
    Returns data with masked values objectively interpolated except where mask==0.

    Arguments:
    data - np.ma.array with mask==True where there is missing data or land.
    mask - np.array of 0 or 1, 0 for land, 1 for ocean.

    Returns a np.ma.array.
    """
    nj, ni = idata.shape
    fdata = idata.filled(
        0.0
    )  # Working with an ndarray is faster than working with a masked array
    if debug:
        plt.figure()
        plt.pcolormesh(mask)
        plt.title("mask")
        plt.colorbar()
        # plt.figure(); plt.pcolormesh(idata.mask); plt.title('idata.mask'); plt.colorbar();
        plt.figure()
        plt.pcolormesh(idata)
        plt.title("idata")
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(idata.filled(3.0))
        plt.title("idata.filled")
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(idata.filled(3.0))
        plt.title("fdata")
        plt.colorbar()
    missing_j, missing_i = np.where(idata.mask & (mask > 0))
    n_missing = missing_i.size
    if verbose:
        print(
            "Data shape: %i x %i = %i with %i missing values"
            % (nj, ni, nj * ni, np.count_nonzero(idata.mask))
        )
        print(
            "Mask shape: %i x %i = %i with %i land cells"
            % (
                mask.shape[0],
                mask.shape[1],
                np.prod(mask.shape),
                np.count_nonzero(1 - mask),
            )
        )
        print("Data has %i missing values in ocean" % (n_missing))
        print("Data range: %g .. %g " % (idata.min(), idata.max()))
    # ind contains column of matrix/row of vector corresponding to point [j,i]
    ind = np.zeros(fdata.shape, dtype=int) - int(1e6)
    ind[missing_j, missing_i] = np.arange(n_missing)
    if verbose:
        print("Building matrix")
    A = scipy.sparse.lil_matrix((n_missing, n_missing))
    b = np.zeros((n_missing))
    ld = np.zeros((n_missing))
    A[range(n_missing), range(n_missing)] = 0.0
    if verbose:
        print("Looping over cells")
    for n in range(n_missing):
        j, i = missing_j[n], missing_i[n]
        im1 = (i + ni - 1) % ni
        ip1 = (i + 1) % ni
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, nj - 1)
        if j > 0 and mask[jm1, i] > 0:
            ld[n] -= 1.0
            ij = ind[jm1, i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[jm1, i]
        if mask[j, im1] > 0:
            ld[n] -= 1.0
            ij = ind[j, im1]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, im1]
        if mask[j, ip1] > 0:
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
        if j == nj - 1 and mask[j, ni - 1 - i] > 0:  # Tri-polar fold
            ld[n] -= 1.0
            ij = ind[j, ni - 1 - i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, ni - 1 - i]
    if debug:
        tmp = np.zeros((nj, ni))
        tmp[missing_j, missing_i] = b
        plt.figure()
        plt.pcolormesh(tmp)
        plt.title("b (initial)")
        plt.colorbar()
    # Set leading diagonal
    b[ld >= 0] = 0.0
    A[range(n_missing), range(n_missing)] = ld - stabilizer
    if debug:
        tmp = np.zeros((nj, ni))
        tmp[missing_j, missing_i] = b
        plt.figure()
        plt.pcolormesh(tmp)
        plt.title("b (final)")
        plt.colorbar()
        tmp = np.ones((nj, ni))
        tmp[missing_j, missing_i] = A.diagonal()
        plt.figure()
        plt.pcolormesh(tmp)
        plt.title("A[i,i]")
        plt.colorbar()
    if verbose:
        print("Matrix constructed")
    A = scipy.sparse.csr_matrix(A)
    if verbose:
        print("Matrix converted")
    new_data = np.ma.array(fdata, mask=(mask == 0))
    if maxiter is None:
        x, info = scipy.sparse.linalg.bicg(A, b)
    elif maxiter == 0:
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        x, info = scipy.sparse.linalg.bicg(A, b, maxiter=maxiter)
    if verbose:
        print("Matrix inverted")
    new_data[missing_j, missing_i] = x
    return new_data
