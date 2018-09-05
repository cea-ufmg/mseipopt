"""Bare wrapper around the IPOPT c interface using ctypes."""


import ctypes
import os
from ctypes import c_int, c_double, c_void_p, CFUNCTYPE, POINTER


_ipopt_lib = None
"""library ctypes.CDLL object or None (if not loaded)."""


c_double_p = POINTER(c_double)
"""Pointer to double."""


c_int_p = POINTER(c_int)
"""Pointer to int."""


Eval_F_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, 
                      c_double_p, c_void_p)
"""Type of the callback for evaluating the objective function."""


Eval_Grad_F_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, 
                      c_double_p, c_void_p)
"""Type of the callback for evaluating the gradient of the objective."""


Eval_G_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, 
                      c_int, c_double_p, c_void_p)
"""Type of the callback for evaluating the constraint function."""


Eval_Jac_G_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, 
                          c_int, c_int,
                          c_int_p, c_int_p, c_double_p,
                          c_void_p)
"""Type of the callback for evaluating the Jacobian of the constraint."""


Eval_H_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, c_double,
                      c_int, c_double_p, c_int,
                      c_int, c_int_p, c_int_p,
                      c_double_p, c_void_p)
"""Type of the callback for evaluating the Hessian of the Lagrangian."""


Intermediate_CB = CFUNCTYPE(c_int, c_int,
                            c_int, c_double,
                            c_double, c_double,
                            c_double, c_double,
                            c_double,
                            c_double, c_double,
                            c_int, c_void_p)
"""Type of the callback to give intermediate execution control to the user."""


def default_ipopt_library_name():
    if os.name == 'nt':
        return "ipopt"
    else:
        return "libipopt.so"


def load_library(name=None):
    global _ipopt_lib
    name = name or default_ipopt_library_name()
    _ipopt_lib = ctypes.cdll.LoadLibrary(name)
    _setup_library()


def _setup_library():
    assert _ipopt_lib is not None, "cannot setup before loading"

    _ipopt_lib.CreateIpoptProblem.restype = c_void_p
    _ipopt_lib.CreateIpoptProblem.argtypes = [
        c_int, c_double_p, c_double_p,
        c_int, c_double_p, c_double_p,
        c_int, c_int, c_int, 
        Eval_F_CB, Eval_G_CB, Eval_Grad_F_CB, Eval_Jac_G_CB, Eval_H_CB
    ]

    _ipopt_lib.FreeIpoptProblem.restype = None
    _ipopt_lib.FreeIpoptProblem.argtypes = [c_void_p]


def default_setup():
    if _ipopt_lib is None:
        load_library()


def CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
                       index_style, eval_f, eval_g, eval_grad_f, eval_jac_g,
                       eval_h):
    default_setup()
    return _ipopt_lib.CreateIpoptProblem(
        n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, index_style,
        eval_f, eval_g, eval_grad_f,  eval_jac_g, eval_h
    )


def AddIpoptStrOption(problem, keyword, val):
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    if isinstance(val, str):
        val = val.encode('ascii')
    return _ipopt_lib.AddIpoptStrOption(problem, keyword, val)


def AddIpoptNumOption(problem, keyword, val):
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    return _ipopt_lib.AddIpoptNumOption(problem, keyword, val)


def AddIpoptIntOption(problem, keyword, val):
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode('ascii')
    return _ipopt_lib.AddIpoptIntOption(problem, keyword, val)
