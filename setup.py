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
    python_requires='==3.10',
    install_requires=[
        "setuptools>=69.0,<69.1",
        "numpy>=1.26,<1.27",
        "xarray>=2023.12,<2024",
        "matplotlib>3.6,<3.7",
        "scipy>=1.11,<1.12",
        "netcdf4>=1.6,<1.7",
        "jupyterlab>=4.0,<4.1",
        "ipympl>=0.9,<0.9.3",
        "sphinx>=5.0,<5.1",
        "black>=24.1,<24.2",
        "pytest>=8.0"
    ]
)
