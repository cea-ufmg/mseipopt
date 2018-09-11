import numpy as np

from mseipopt import bare, bare_np, ez


def f(x):
    return (x[0] - 1)**2


def grad(x):
    return [2*(x[0] - 1)]


def g(x):
    return []


def jac_ind():
    return [[], []]


def jac_val(x):
    return []


def hess_ind():
    return [[0], [0]]


def hess_val(x, obj_factor, mult):
    return [2 * obj_factor]


if __name__ == '__main__':
    x_L = [-100]
    x_U = [100]
    x_b = (x_L, x_U)
    g_b = ([], [])
    jac = jac_ind, jac_val
    h = hess_ind, hess_val
    with ez.Problem(x_b, g_b, f, g, grad, jac, 0, h, 1) as problem:
        problem.add_str_option('linear_solver', 'mumps')
        problem.add_int_option('max_iter', 2000)
        problem.add_num_option('tol', 1e-6)
        
        x0 = [10.0]
        xopt, info = problem.solve(x0)
