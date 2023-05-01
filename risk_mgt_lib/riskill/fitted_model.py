import numpy as np
from scipy import optimize
import scipy.stats as sps


def OLS(X, Y):
    _X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    B = np.matrix(_X.T @ _X).I @ _X.T @ Y
    e = Y - _X@B
    return np.array(B), e


def fit_regression_t(X, Y):
    def negative_t_ll(arguments):
        s, free, beta, intercept= arguments[0], arguments[1], arguments[2], arguments[3]
        er = Y - X * beta - intercept
        ll = sps.t.logpdf(er, free, loc=0.0, scale=s).sum()
        return -ll

    start_beta = OLS(X, Y)[0]
    start_c = start_beta[0]
    start_b = start_beta[1]
    e = Y - X * start_b - start_c
    start_nu = 6.0/sps.kurtosis(e) + 4.
    start_s = np.sqrt(np.var(e)*(start_nu-2)/start_nu)
    print(start_c, start_b, start_nu, start_s)

    cons = ({'type': 'ineq', 'fun': lambda arguments: arguments[0] - 1e-6},
            {'type': 'ineq', 'fun': lambda arguments: arguments[1] - 2.0})
    if start_nu > 2.0:
        x0 = np.array([0.1, start_nu, start_b, start_c]).reshape(-1)
    else:
        x0 = np.array([0.1, 11.5, start_b, start_c]).reshape(-1)
    result_t = optimize.minimize(negative_t_ll, x0, constraints=cons)
    t_beta = result_t.x[2]
    t_intercept = result_t.x[3]
    t_free = result_t.x[1]
    t_s = result_t.x[0]
    t_ll = -negative_t_ll([t_s, t_free, t_beta, t_intercept])
    AIC = 2 * 4 - 2 * t_ll

    dic = {'t_intercept': t_intercept, 't_beta': t_beta, 't_free': t_free,
           't_s': t_s, 't_ll': t_ll, 'AIC': AIC}

    return dic
