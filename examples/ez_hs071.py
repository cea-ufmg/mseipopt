"""
Example of IPOPT interface for solving problem 71 of the Hock--Schittkowsky
test suite.

W. Hock and K. Schittkowski. 
Test examples for nonlinear programming codes. 
Lecture Notes in Economics and Mathematical Systems, 187, 1981. 
doi: 10.1007/978-3-642-48320-2.
"""


import numpy as np

from mseipopt import bare, bare_np, ez


def f(x):
    return x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]


def grad(x):
    return [x[0]*x[3] + x[3]*(x[0] + x[1] + x[2]),
            x[0]*x[3],
            x[0]*x[3] + 1.0,
            x[0]*(x[0] + x[1] + x[2])]

def g(x):
    return [x[0]*x[1]*x[2]*x[3], 
            x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]]


def jac_ind():
    return ([0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 2, 3, 0, 1, 2, 3])


def jac_val(x):
    return [x[1]*x[2]*x[3], 
            x[0]*x[2]*x[3], 
            x[0]*x[1]*x[3], 
            x[0]*x[1]*x[2],
            2.0*x[0],
            2.0*x[1],
            2.0*x[2],
            2.0*x[3]]


def hess_ind():
    return ([0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])


def hess_val(x, obj_factor, mult):
    return [obj_factor*2*x[3] + mult[1]*2,
            obj_factor*x[3] + mult[0]*(x[2]*x[3]),
            mult[1]*2,
            obj_factor*(x[3]) + mult[0]*(x[1]*x[3]),
            mult[0]*(x[0]*x[3]),
            mult[1]*2,
            obj_factor*(2*x[0] + x[1] + x[2]) +  mult[0]*x[1]*x[2],
            obj_factor*x[0] + mult[0]*x[0]*x[2],
            obj_factor*x[0] + mult[0]*x[0]*x[1],
            mult[1]*2]


if __name__ == '__main__':
    x_L = [1.0] * 4
    x_U = [5.0] * 4
    x_b = (x_L, x_U)
    g_b = ([25, 40], [2e19, 40])
    jac = jac_ind, jac_val
    h = hess_ind, hess_val
    with ez.Problem(x_b, g_b, f, g, grad, jac, 8, h, 10) as problem:
        x0 = [1, 5, 5, 1]
        xopt, info = problem.solve(x0)
    
    expected_xopt = np.r_[1, 4.743, 3.82115, 1.379408]
    err = xopt - expected_xopt
