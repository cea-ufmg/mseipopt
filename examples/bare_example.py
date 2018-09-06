import ctypes
import imp
from ctypes import c_double, pointer, POINTER

import numpy as np

from mseipopt import bare


@bare.Eval_F_CB
def eval_f(n, x, new_x, obj_value, user_data):
    obj_value[0] = (x[0] - 1)**2
    return 1


@bare.Eval_Grad_F_CB
def eval_grad_f(n, x, new_x, grad_f, user_data):
    grad_f[0] = 2*(x[0] - 1)
    return 1


@bare.Eval_G_CB
def eval_g(n, x, new_x, m, g, user_data):
    return 1


@bare.Eval_Jac_G_CB
def eval_jac_g(n, x, new_x, m, nele_jac, iRow, jCol, values, user_data):
    return 1


@bare.Eval_H_CB 
def eval_h(n, x, new_x, obj_factor, m, mult, new_mult, nele_hess,
           iRow, jCol, values, user_data):
    if iRow:
        iRow[0] = 0
    if jCol:
        jCol[0] = 0
    if values:
        values[0] = 2 * obj_factor
    return 1


if __name__ == '__main__':
    x_L = c_double(-100)
    x_U = c_double(100)
    problem = bare.CreateIpoptProblem(
        1, pointer(x_L), pointer(x_U),
        0, None, None, 0, 1, 0, 
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
    )
    bare.AddIpoptStrOption(problem, 'linear_solver', 'mumps')
    bare.AddIpoptIntOption(problem, 'max_iter', 2000)
    bare.AddIpoptNumOption(problem, 'tol', 1e-6)
    
    x = c_double(10)
    obj_val = c_double()
    status = bare.IpoptSolve(problem, pointer(x), None, pointer(obj_val), 
                             None, None, None, None)
    bare.FreeIpoptProblem(problem)
