import ctypes
import imp
from ctypes import c_double, pointer, POINTER

import numpy as np

from mseipopt import bare, bare_np


def f(x, new_x, obj_value):
    obj_value[...] = (x[0] - 1)**2
    return 1


def grad_f(x, new_x, grad_f):
    grad_f[0] = 2*(x[0] - 1)
    return 1


def g(x, new_x, g):
    return 1


def jac_g(x, new_x, iRow, jCol, values):
    return 1


def h(x, new_x, obj_factor, mult, new_mult, iRow, jCol, values):
    if iRow is not None:
        iRow[0] = 0
    if jCol is not None:
        jCol[0] = 0
    if values is not None:
        values[0] = 2 * obj_factor
    return 1


if __name__ == '__main__':
    x_L = [-100]
    x_U = [100]
    x_b = (x_L, x_U)
    g_b = ([], [])
    with bare_np.Problem(x_b, g_b, 0, 1, 0, f, g, grad_f, jac_g, h) as problem:
        problem.add_str_option('linear_solver', 'mumps')
        problem.add_int_option('max_iter', 2000)
        problem.add_num_option('tol', 1e-6)
        
        x = np.array([10.0])
        obj_val = np.empty(())
        status = problem.solve(x, None, obj_val, None, None, None)
