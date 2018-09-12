"""
Test of IPOPT interface with problem 71 of the Hock--Schittkowsky test suite.
"""


import numpy as np

from mseipopt import ez


def test_hs071():
    def f(x):
        """Objective function."""
        return x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
    
    def grad(x):
        """Gradient of objective function."""
        return [x[0]*x[3] + x[3]*(x[0] + x[1] + x[2]),
                x[0]*x[3],
                x[0]*x[3] + 1.0,
                x[0]*(x[0] + x[1] + x[2])]
    
    def g(x):
        """Constraint function."""
        return [x[0]*x[1]*x[2]*x[3], 
                x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]]

    def jac_ind():
        """Indices of constraint Jacobian elements."""
        return ([0, 0, 0, 0, 1, 1, 1, 1],
                [0, 1, 2, 3, 0, 1, 2, 3])


    def jac_val(x):
        """Values of nonzero constraint Jacobian elements."""
        return [x[1]*x[2]*x[3], 
                x[0]*x[2]*x[3], 
                x[0]*x[1]*x[3], 
                x[0]*x[1]*x[2],
                2.0*x[0],
                2.0*x[1],
                2.0*x[2],
                2.0*x[3]]
    
    def hess_ind():
        """Indices of Lagrangian Hessian elements."""
        return ([0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
                [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])

    def hess_val(x, obj_factor, mult):
        """Values of nonzero Lagrangian Hessian  elements."""
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

    x_b = ([1.0] * 4, [5.0] * 4)
    g_b = ([25, 40], [2e19, 40])
    jac = jac_ind, jac_val
    h = hess_ind, hess_val
    with ez.Problem(x_b, g_b, f, g, grad, jac, 8, h, 10) as problem:
        x0 = [1, 5, 5, 1]
        xopt, info = problem.solve(x0)
    
    expected_xopt = np.r_[1, 4.743, 3.82115, 1.379408]
    err = xopt - expected_xopt
    np.testing.assert_almost_equal(xopt, expected_xopt, decimal=6)
