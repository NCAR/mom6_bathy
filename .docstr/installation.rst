Installation
======================================

Prerequisite
-------------
* Python 3.7+: Make sure that you have Python 3.7 (or a later version) installed. 

Note: If you are familiar with conda or Python virtual environments, it is strongly
advised, but not required, to work on a new virtual environment dedicated to 
the use of the mom6_bathy tool only. In conda framework, for example:

.. code-block:: bash

    # create a new conda environment (execute one time only)
    conda create --name mom6_bathy_env

    # activate this new environment (before installing mom6_bathy
    # and before each use session.)
    conda activate mom6_bathy_env

Instructions
-------------

First, clone the `mom6_bathy` GitHub repository as follows:

.. code-block:: bash

    git clone --recursive https://github.com/NCAR/mom6_bathy.git

Then, `cd` into your newly checked out `mom6_bathy` clone and run the
installation script as follows:

.. code-block:: bash

    cd mom6_bathy
    python setup.py build
    python setup.py install

To confirm that the installation was successful, execute the following command:

.. code-block:: bash

    python -c "import mom6_bathy"

If no error message is displayed, then the installation is successful.
