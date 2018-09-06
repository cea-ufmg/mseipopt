"""High-Level, pythonic, safe and easy IPOPT interface."""


import ctypes
import functools

import numpy as np

from . import bare


class Problem:
    def __init__(self, x_bounds, g_bounds, jac_ind, hess_ind, 
                 f, g, grad_f, jac_g, h=None, index_style=0):
        x_L, x_U = x_bounds
        x_L = np.asarray(x_L)
        x_U = np.asarray(x_U)
        n = x_L.size
        if x_U.size != n:
            raise ValueError("Inconsistent sizes of 'x' lower and upper bounds")
        
        g_L, g_U = g_bounds
        g_L = np.asarray(g_L)
        g_U = np.asarray(g_U)
        m = g_L.size
        if g_U.size != m:
            raise ValueError("Inconsistent sizes of 'g' lower and upper bounds")
        
        eval_f = wrap_f(f)
        eval_grad_f = wrap_grad_f(grad_f)
        eval_g = wrap_g(g)

        problem = bare.CreateIpoptProblem(
            n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
            index_style, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
        )
        if not problem:
            raise RuntimeError('Error creating IPOPT problem')
        
        self._problem = problem
        """Pointer to the underlying `IpoptProblemInfo` structure."""
    
    def free(self):
        if not self._problem:
            raise RuntimeError('Problem invalid or already freed')
        bare.FreeIpoptProblem(self._problem)
        self._problem = None


def wrap_f(f):
    @functools.wraps(f)
    @bare.Eval_F_CB
    def wrapper(n, x, new_x, obj_value, user_data):
        x_array = np.ctypeslib.as_array(x, (n,))
        obj_value[0] = f(x_array)
    return wrapper


def wrap_grad_f(grad_f):
    @functools.wraps(grad_f)
    @bare.Eval_Grad_F_CB
    def wrapper(n, x, new_x, grad_value, user_data):
        x_array = np.ctypeslib.as_array(x, (n,))
        grad_f_array = np.ctypeslib.as_array(grad_value, (n,))
        grad_f_array[...] = grad_f(x_array)
    return wrapper


def wrap_g(g):
    @functools.wraps(g)
    @bare.Eval_G_CB
    def wrapper(n, x, new_x, m, g, user_data):
        x_array = np.ctypeslib.as_array(x, (n,))
        g_array = np.ctypeslib.as_array(grad_value, (m,))
        g_array[...] = g(x_array)
    return wrapper
