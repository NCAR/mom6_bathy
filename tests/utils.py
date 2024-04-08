'''Functions that are used in tests.'''

import socket

def on_cisl_machine():
    '''Return True if the current machine is a CISL machine, False otherwise.'''
    hostname = socket.gethostname()
    return ('derecho' in hostname or 'casper' in hostname)