"""High-Level, pythonic, safe and easy IPOPT interface."""


import functools
import inspect

import numpy as np

from . import bare_np


class UnmanagedProblem(bare_np.Problem):
    def solve(self, x, g_mult=None, x_L_mult=None, x_U_mult=None):
        pass


class Problem(bare_np.Problem):
    def __init__(self, x_bounds, g_bounds, f, g, grad,
                 jac, nele_jac, hess=None, nele_hess=None):
        if hess is not None and nele_hess is None:
            raise TypeError("'nele_hess' must be given if 'hess' is supplied")
        if hess is None:
            nele_hess = 0
        f_cb = f_callback(f)
        grad_cb = grad_callback(grad)
        g_cb = g_callback(g)
        jac_cb = jac_callback(jac)
        hess_cb = hess_callback(hess)
        self._create_args = (x_bounds, g_bounds, nele_jac, nele_hess, 0,
                             f_cb, g_cb, grad_cb, jac_cb, hess_cb)
        """Arguments for problem creation."""
        
        self._problem = None
        """Underlying problem object."""
    
    def __enter__(self):
        if self._problem:
            raise RuntimeError("entering an opened context")
        self._problem = bare_np.Problem(*self._create_args)
        return self._problem.__enter__()
        
    def __exit__(self, exc_type, exc_value, traceback):
        if not self._problem:
            raise RuntimeError("attempt to exit from invalid context")
        self._problem.__exit__(exc_type, exc_value, traceback)
        self._problem = None


def f_callback(f):
    @functools.wraps(f)
    def wrapper(x, new_x, obj_value):
        obj_value[()] = f(x)
        return 1
    return wrapper



def grad_callback(grad):
    @functools.wraps(grad)
    def wrapper(x, new_x, grad_array):
        grad_array[()] = grad(x)
        return 1
    return wrapper


def g_callback(g):
    def wrapper(x, new_x, g_array):
        if g_array.size:
            g_array[()] = g(x)
        return 1
    return wrapper


def jac_callback(jac):
    jac_ind, jac_val = jac
    def wrapper(x, new_x, iRow, jCol, values):
        # Fill out Jacobian values
        if values is not None:
            if values.size == 0:
                return 1
            if accepts_output(jac_val):
                jac_val(x, out=values)
            else:
                values[...] = jac_val(x)
            return 1
        
        # Fill out jacobian indices
        i, j = jac_ind() if callable(jac_ind) else jac_ind
        if iRow is not None and iRow.size:
            iRow[...] = i
        if jCol is not None and jCol.size:
            jCol[...] = j
        return 1
    
    return wrapper


def hess_callback(hess):
    if hess is None:
        return
    
    hess_ind, hess_val = hess    
    def wrapper(x, new_x, obj_factor, mult, new_mult, iRow, jCol, values):
        # Fill out Hessian values
        if values is not None:
            if values.size == 0:
                return 1
            if accepts_output(hess_val):
                hess_val(x, out=values)
            else:
                values[...] = hess_val(x, obj_factor, mult)
            return 1
        
        # Fill out jacobian indices
        i, j = hess_ind() if callable(hess_ind) else hess_ind
        if iRow is not None and iRow.size:
            iRow[...] = i
        if jCol is not None and jCol.size:
            jCol[...] = j
        return 1
    
    return wrapper


@functools.lru_cache()
def accepts_output(f):
    params = inspect.signature(f).parameters
    out = params.get('out', None)
    if out is None:
        return False

    # First parameter cannot be the output
    if list(params).index('out') == 0:
        return False
    
    kinds = inspect.Parameters
    return out.kind == kinds.POSITIONAL_OR_KEYWORD or kinds.KEYWORD_ONLY
            
