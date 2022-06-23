import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mom6_bathy", # Replace with your own username
    version="0.0.1",
    author="Alper Altuntas",
    author_email="altuntas@ucar.edu",
    description="MOM6 simple grid and bathymetry generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NCAR/mom6-bathy",
    packages=['mom6_bathy', 'midas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "setuptools>=62.5",
        "numpy>=1.19",
        "xarray>=0.16",
        "matplotlib>=3.3.0",
        "scipy>=1.5.1",
        "netCDF4",
        "jupyterlab",
        "ipympl",
    ]
)
