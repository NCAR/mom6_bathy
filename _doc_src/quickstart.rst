Quickstart Guide
======================================

mom6_tools library can be utilized via its Python API, i.e., directly in Python
scripts or in Jupyter notebooks. In this quickstart guide, we describe how
the tool can be utilized within a Jupyter Notebook, but the majority of
these instructions apply to Python scripts as well.

Step 1: Import modules
----------------------------------------------

The first step is to import some relevant, auxiliary Python libraries. (These
libraries are automatically installed when the mom6_bathy installer gets
executed.)

.. code-block:: python

    import os
    import numpy as np
    import matplotlib.pyplot as plt

We then import the ``mom6grid`` class of the mom6_bathy package as follows.
This class is used to generate a horizontal grid, which is the first
step of constructing a MOM6 grid and an associated bathymetry:

.. code-block:: python

    from mom6_bathy.mom6grid import mom6grid

Step 2: Create the horizontal grid instance
-------------------------------------------

After having imported the modules, we can now create a horizontal grid.
An exmaple mom6grid object instantiation:

.. code-block:: python

    grd = mom6grid(
        nx         = 180,           # Number of grid points in x direction
        ny         = 90,            # Number of grid points in y direction
        config     = "cartesian",   # Grid configuration. Valid values: 'cartesian', 'mercator', 'spherical'
        axis_units = "degrees",     # Grid axis units. Valid values: 'degrees', 'm', 'km'
        lenx       = 360.0,         # grid length in x direction, e.g., 360.0 (degrees)
        leny       = 160.0,         # grid length in y direction
        ystart     = -80.0          # starting y coordinate
    )

In the above example, the ``mom6grid`` object, named ``grd``, is constructed by
specifying the required argumants ``nx``, ``ny``, ``config``, ``axis_units``,  ``lenx``,
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
supergrid......

