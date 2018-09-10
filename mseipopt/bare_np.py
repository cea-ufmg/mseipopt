"""Bare wrapper around IPOPT using numpy arrays and exception handling."""


import numpy as np
from numpy.ctypeslib import as_array

from . import bare


class Problem:
    def __init__(self, x_bounds, g_bounds, nele_jac, nele_hess, index_style,
                 f, g, grad_f, jac_g, h=None):
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
        eval_g = wrap_g(g)
        eval_grad_f = wrap_grad_f(grad_f)
        eval_jac_g = wrap_jac_g(jac_g)
        eval_h = wrap_h(h) if h is not None else h
        problem = bare.CreateIpoptProblem(
            n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
            index_style, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
        )
        if not problem:
            raise RuntimeError('Error creating IPOPT problem')
        self._problem = problem
        """Pointer to the underlying `IpoptProblemInfo` structure."""
        
        if h is None:
            self.str_option( 'hessian_approximation', 'limited-memory')
    
    def free(self):
        if not self._problem:
            raise RuntimeError('Problem invalid or already freed')
        bare.FreeIpoptProblem(self._problem)
        self._problem = None
    
    def add_str_option(self, keyword, val):
        if not bare.AddIpoptStrOption(self._problem, keyword, val):
            raise ValueError(f'invalid option or value')

    def add_int_option(self, keyword, val):
        if not bare.AddIpoptIntOption(self._problem, keyword, val):
            raise ValueError(f'invalid option or value')

    def add_num_option(self, keyword, val):
        if not bare.AddIpoptNumOption(self._problem, keyword, val):
            raise ValueError(f'invalid option or value')
        
    def __enter__(self):
        if not self._problem:
            raise RuntimeError('Invalid context or reentering context.')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.free()


def wrap_f(f, handler=default_handler):
    @functools.wraps(f)
    @bare.Eval_F_CB
    def wrapper(n, x, new_x, obj_value, user_data):
        try:
            x_array = as_array(x, n)
            obj_value_array = as_array(obj_value, ())
            return f(x_array, new_x, obj_value_array)
        except Exception as e:
            if callable(handler):
                handler(e)
            return 0
    return wrapper


def wrap_grad_f(grad_f, handler=default_handler):
    @functools.wraps(grad_f)
    @bare.Eval_Grad_F_CB
    def wrapper(n, x, new_x, grad_value, user_data):
        try:
            x_array = as_array(x, n)
            grad_f_array = as_array(grad_value, n)
            return grad_f(x_array, new_x, grad_f_array)
        except Exception as e:
            if callable(handler):
                handler(e)
            return 0        
    return wrapper


def wrap_g(g, handler=default_handler):
    @functools.wraps(g)
    @bare.Eval_G_CB
    def wrapper(n, x, new_x, m, g, user_data):
        try:
            x_array = as_array(x, n)
            g_array = as_array(grad_value, m)
            return g(x_array, new_x, g_array)
        except Exception as e:
            if callable(handler):
                handler(e)
            return 0
    return wrapper


def wrap_jac_g(jac_g, handler=default_handler):
    @functools.wraps(jac_g)
    @bare.Eval_Jac_G_CB
    def wrapper(n, x, new_x, m, nele_jac, iRow, jCol, values, user_data):
        try:
            x_array = as_array(x, n) if x else None
            i_array = as_array(iRow, nele_jac) if iRow else None
            j_array = as_array(jCol, nele_jac) if jCol else None
            values_array = as_array(values, nele_jac) if val else None
            return jac_g(x_array, new_x, i_array, j_array, values_array)
        except Exception as e:
            if callable(handler):
                handler(e)
            return 0
    return wrapper


def wrap_h(h, handler=default_handler):
    @functools.wraps(h)
    @bare.Eval_H_CB 
    def wrapper(n, x, new_x, obj_factor, m, mult, new_mult, nele_hess,
                iRow, jCol, values, user_data):
        try:
            x_array = as_array(x, n) if x else None
            obj_factor_array = as_array(obj_factor, ()) if obj_factor else None
            mult_array = as_array(mult, m) if mult else None
            i_array = as_array(iRow, nele_hess) if iRow else None
            j_array = as_array(jCol, nele_hess) if jCol else None
            values_array = as_array(values, nele_hess) if values else None
            return h(x_array, new_x, obj_factor_array, mult_array, new_mult,
                     i_array, j_array, val_array)
        except Exception as e:
            if callable(handler):
                handler(e)
            return 0


def default_handler(e):
    """Exception handler for IPOPT ctypes callbacks, prints the traceback."""
    import traceback
    traceback.print_exc()
