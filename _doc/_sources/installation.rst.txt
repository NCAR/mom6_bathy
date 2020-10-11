Installation
======================================

Prerequisite
-------------
    Python 3.7+

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
