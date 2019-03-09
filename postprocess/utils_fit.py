
import numpy as np

# poly parameters start from highest order to lowerest order
def jacobian(x, order):
        return np.array([x**(ii) for ii in range(order, -1, -1)])

def poly_fit(X, Y, order):
    pp, vv = np.polyfit(X, Y, order, cov=True)
    f_y, f_y_e = poly_val(pp, vv, X)
    chi2 = np.sum((f_y-Y)**2/f_y_e**2)
    return pp, vv, chi2

def poly_val(pp, vv, X):
    f_y   = np.polyval(pp, X)
    f_y_e = np.array([np.sqrt(np.matmul(jacobian(x, order), np.matmul(vv, jacobian(x, order).transpose()))) for x in X])
    return f_y, f_y_e

def poly_fit2(X, Y, order):
    pp = np.polyfit(X, Y, order)
    f_y = np.polyval(pp, X)
    diff = np.sum(np.sqrt((f_y - Y)**2/Y**2))
    return pp, f_y, diff


def poly_fit_phi(X, Y):
    """X is z, Y is phi"""
    # phi is constrained to [-pi, pi]
    order = 1
    pp = np.polyfit(X, Y, order)
    f_y = np.polyval(pp, X)
    diff = np.sum(np.sqrt((f_y - Y)**2/Y**2))
    return pp, f_y, diff
