Quickstart Guide
======================================

`mom6_tools` library can be utilized via its Python API, i.e., directly within Python
scripts or within Jupyter notebooks. In this quickstart guide, we describe how
the tool can be utilized within a Jupyter Notebook, but the majority of
these instructions apply to Python scripts as well.

Step 1: Import modules
----------------------------------------------

The first step is to import the ``mom6grid`` and ``mom6bathy`` classes of the 
`mom6_bathy` package, in addition to ``numpy``. The ``mom6grid`` class is used
to generate a horizontal grid, which is the first step of constructing a MOM6
grid. The ``mom6bathy`` class, on the other hand, is used to generate an 
associated bathymetry. Note that our Python package containing both of these 
classes is called `mom6_bathy`.

.. code-block:: python

    import numpy as np
    from mom6_bathy.mom6grid import mom6grid
    from mom6_bathy.mom6bathy import mom6bathy

Step 2: Create the horizontal grid 
-------------------------------------------

After having imported the modules, we can now create a horizontal grid.
An exmaple mom6grid object instantiation:

.. code-block:: python

    grd = mom6grid(
        nx         = 180,           # Number of grid points in x direction
        ny         = 90,            # Number of grid points in y direction
        config     = "spherical",   # Grid configuration. Valid values: 'cartesian', 'mercator', 'spherical'
        axis_units = "degrees",     # Grid axis units. Valid values: 'degrees', 'm', 'km'
        lenx       = 360.0,         # grid length in x direction, e.g., 360.0 (degrees)
        leny       = 160.0,         # grid length in y direction
        ystart     = -80.0          # starting y coordinate
    )

In the above example, the ``mom6grid`` object, named ``grd``, is constructed by
specifying the required arguments ``nx``, ``ny``, ``config``, ``axis_units``,  ``lenx``,
and ``leny``, in addition to the optional argument ``ystart``. The full list of 
``mom6grid`` arguments and their descriptions may be printed by running
``mom6grid?`` statement on a notebook cell:


.. code-block::

    mom6grid?

    ...

    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    config : str or None
        Grid configuration. Valid values: 'cartesian', 'mercator', 'spherical'
    axis_units : str
        Grid axis units. Valid values: 'degrees', 'm', 'km'
    lenx : float
        grid length in x direction, e.g., 360.0 (degrees)
    leny : float
        grid length in y direction, e.g., 160.0 (degrees)
    srefine : int, optional
        refinement factor for the supergrid. 2 by default
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

Notice above that ``mom6grid`` class may be used to create 'cartesian', 'mercator',
or, 'spherical' grids, as well as tripolar or displaced pole grids. 

*Avoiding singularity points*
*****************************

To avoid singularity points within the ocean grid:
  * The grid poles (which may be different than the true poles) must be left out of the grid,
    by making sure that the extent of the grid in the y-direction do not cover the poles, 
    e.g., by setting ``ystart`` to -80.0 degrees
    and ``leny`` to 160.0 degrees.
  * Alternatively, one or two singularities (typically, in the northern hemisphere) may be 
    displaced into land masses if ``displace_pole`` or ``tripolar_n`` options are to be used.
    The other singularity (typically, in the southern hemisphere) would still need to be
    left out the geographic extent of the grid.

In either case, a land component (active, data, or stub) must be present for the purpose of
hiding the singularity points of spherical ocean grids within the CESM framework.

*Grid Metrics and Attributes*
*****************************

When a ``mom6grid`` grid instance gets created, several grid metrics and attributes
(on all staggerings) are automatically computed and populated. These metrics and attributes
are accessible via the accessor operator (``.``). For example, to access "the array
of t-grid longitutes" of ``grd``:
    
.. code-block:: python

    grd.tlon

The full list of grid metrics and attributes:

* ``tlon``: array of t-grid longitudes
* ``tlat``: array of t-grid latitudes
* ``ulon``: array of u-grid longitudes
* ``ulat``: array of u-grid latitudes
* ``vlon``: array of v-grid longitudes
* ``vlat``: array of v-grid latitudes
* ``qlon``: array of corner longitudes
* ``qlat``: array of corner latitudes
* ``dxt``: x-distance between U points, centered at t
* ``dyt``: y-distance between V points, centered at t
* ``dxCv``: x-distance between q points, centered at v
* ``dyCu``: y-distance between q points, centered at u
* ``dxCu``: x-distance between y points, centered at u
* ``dyCv``: y-distance between t points, centered at v
* ``angle``: angle T-grid makes with latitude line
* ``tarea``: T-cell area


*Supergrid*
*****************************
In addition to above grid metrics and attributes, the ``mom6grid`` class incorporates an
underlying :term:`supergrid` instance associated the grid instance, which is again
accessible via the (``.``) operator:

.. code-block:: python

    grd.supergrid

Any user changes to coordinates, e.g., increasing the equatorial resolution,
must be applied to the supergrid, using the ``update_supergrid`` method. This is
because the supergrid is the underlying refined grid that is used to determine the
the four staggered grids (T,U,V,Q) that forms the actual computational grid.
Users can modify the supergrid by providing a new x and y coordinate arrays, e.g.,
as follows:

.. code-block:: python

  grd.update_supergrid(xdat, ydat)

where ``xdat`` and ``ydat`` are user-defined 2-dimensional numpy arrays containing
to the new x and y coordinates of the supergrid. Running the ``update_supergrid``
method of a ``mom6grid`` instance automatically updates all other grid metrics listed
above.

Note: the supergrid implementation in `mom6_bathy` relies on `MIDAS <https://github.com/mjharriso/MIDAS>`_,
a python library developed by M. Harrison (GFDL).

Step 3: Create Bathymetry
----------------------------------------------

After having generated the horizontal grid, we can now create an associated bathymetry
using the ``mom6bathy`` class of the `mom6_bathy` tool. We instantiate a bathymetry 
object as follows:

.. code-block:: python

    bathy = mom6bathy(grd, min_depth=10.0)

The first argument (``grd``) of ``mom6bathy`` constructor is the horizontal grid instance for which
the bathymetry is to be created, while the second argument (``min_depth``) is the minimum ocean depth.
Any column in the ocean grid with a depth shallower than ``min_depth``  is masked out of the ocean
domain. The minimum depth attribute of a bathymetry instance may be changed afterwards using the
assignment operator. For example:

.. code-block:: python

    bathy.min_depth = 5.0

*Predefined Bathymetry Configurations*
**************************************
The ``mom6bathy`` class provides three predefined bathymetry configurations, which are also
available in MOM6 as idealized configurations. (See `TOPO_CONFIG` parameter in MOM_input)

  * `flat`: flat bottom set to MAXIMUM_DEPTH. Example:
  * `bowl`: an analytically specified bowl-shaped basin ranging between MAXIMUM_DEPTH and MINIMUM_DEPTH.
  * `spoon`: a similar shape to 'bowl', but with an vertical wall at the southern face.

Examples:

.. code-block:: python

    # flat bottom
    bathy.set_flat(D=500.0)

    # bowl
    bathy.set_bowl(500.0, 50.0, expdecay=1e7)

    # spoon
    bathy.set_spoon(500.0, 50.0, expdecay=1e7)
    
The first and the second arguments of ``set_bowl`` and ``set_spoon`` methods are maximum depth
and minimum depth, respectively.

*Custom Bathymetry*
*************************************
In addition to the above predefined configurations, users may provide their own depth arrays. For
example:
  
.. code-block:: python

    # define a custom depth
    xi = grd.tlat.nx.data
    yi = grd.tlat.ny.data[:,np.newaxis]
    custom_d = 1000.0 + 50.0 * np.sin(xi*np.pi/6) * np.cos(yi*np.pi/6)

    # update the bathymetry:
    bathy.set_depth_arr(custom_d)

*Adding ridges*
*************************************
Simpler model bathymetry configurations typically include ridges to represent straits and
continents in an idealized manner. The ``mom6bathy`` class provides ``apply_ridge`` method
to add ridges to the bathymetry. Example usage:

.. code-block:: python

  bathy.apply_ridge(height=200, width=8, lon=240, ilat=(10,80) )

See the `Examples` page to access Jupyter notebooks with example usages of these methods.

Step 4: Write NetCDF Grid Files
----------------------------------------------

The final step of `mom6_bathy` workflow is to write out the netcdf files containing grid
and bathymetry data. These files are to be read in by MOM6 during runtime.

*Supergrid File*
****************

``GRID_FILE`` parameter in ``MOM_input`` file is to be set to the supergrid file path,
which is specified as ``./ocean_hgrid.nc`` in the below example.

.. code-block:: python

  grd.to_netcdf(supergrid_path = "./ocean_hgrid.nc")

*Topography (Bathymetry) File*
******************************

``TOPO_FILE`` parameter in ``MOM_input`` file is to be set to the topography file path,
which is specified as ``./ocean_topog.nc`` in the below example.

.. code-block:: python

  bathy.to_topog(supergrid_path = "./ocean_topog.nc")

*SCRIP File and ESMF Mesh file*
*******************************

In addition to the supergrid file and the topography file, which are both MOM6 input files,
the CESM simpler models users are required to generate a SCRIP file and an additional
ESMF mesh file to be able work with the NUOPC driver.

The SCRIP file, which contains the horizontal grid file data in a specific format, is
used within the CESM framework to generate domain and mapping files. This file is needed
when running CESM with the MCT driver. To create a SCRIP file from a `mom6bathy` instance,
execute the ``to_topog`` method. For example:

.. code-block:: python

  bathy.to_SCRIP(SCRIP_path="./SCRIP_mom6_idealized.nc")

If CESM is to be run with the NUOPC driver, however, the users are required to create
an ESMF mesh file, which is a variant of the SCRIP grid file. Users can create an ESMF
mesh file from an existing SCRIP file using the ``ESMF_Scrip2Unstruct`` tool available
from the ESMF toolkit.


.. code-block:: bash

  $ ESMF_Scrip2Unstruct sg0v1_SCRIP.nc sg0v1_ESMFmesh.nc 0



Further steps
----------------------------------------------

The remaining steps of configuring the model, which are listed below, are beyond
the scope of the `mom6_bathy` tool, but we give a brief overview here as a reference.

*Initial Conditions*
*******************************
Users need to generate initial conditions for a newly generated grid and bathymetry.
Depending on the configuration, creating initial conditions may involve mapping and 
regridding of already available initial conditions, or, for simpler configurations, users
may choose to use out-of-the-box MOM6 idealized temperature-salinity profiles as
initial conditions. This may be accomplished by setting the ``TS_COMFIG`` parameter
in ``MOM_input`` file to ``fit``, ``linear``, ``USER``, etc.

*Configuring MOM6*
*******************************

Several ``MOM_input`` (or ``MOM_override``) entries need to be updated for a newly
created grid and bathymetry configuration. To print out the values of these parameters,
one can run the ``print_MOM6_runtime_params()`` method of ``mom6bathy``. The list
of these parameters:

  * TRIPOLAR_N
  * NIGLOBAL
  * NJGLOBAL
  * GRID_CONFIG
  * GRID_FILE
  * TOPO_CONFIG
  * TOPO_FILE
  * MAXIMUM_DEPTH
  * MINIMUM_DEPTH
  * REENTRANT_X

*Configuring CESM*
*******************************
[TODO]

