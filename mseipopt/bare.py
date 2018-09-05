"""Bare wrapper around the IPOPT c interface using ctypes."""


import ctypes
import os



_ipopt_lib = None


def default_ipopt_library_name():
    if os.name == 'nt':
        return "ipopt"
    else:
        return "libipopt.so"


def load_library(name=None):
    global _ipopt_lib
    if _ipopt_lib is None:
        name = name or default_ipopt_library_name()
        _ipopt_lib = ctypes.cdll.LoadLibrary()


def AddIpoptStrOption(problem, keyword, val):
    assert _ipopt_lib is not None, "Library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    if isinstance(val, str):
        val = val.encode('ascii')
    return _ipopt_lib.AddIpoptStrOption(problem, keyword, val)


def AddIpoptNumOption(problem, keyword, val):
    assert _ipopt_lib is not None, "Library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    return _ipopt_lib.AddIpoptNumOption(problem, keyword, val)


def AddIpoptIntOption(problem, keyword, val):
    assert _ipopt_lib is not None, "Library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    return _ipopt_lib.AddIpoptIntOption(problem, keyword, val)

