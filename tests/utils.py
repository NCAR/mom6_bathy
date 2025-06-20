"""Functions that are used in tests."""

import socket


def on_cisl_machine():
    """Return True if the current machine is a CISL machine, False otherwise."""
    fqdn = socket.getfqdn()
    return "hpc.ucar.edu" in fqdn
