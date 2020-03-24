
import numpy as np
from .. import pairwise


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


def correct_phi(phi_list):
    all_pairs = pairwise(phi_list)
    new_list = [ phi_list[0] ]

    offset = 0
    for pp in all_pairs:
        diff = pp[1] - pp[0]
        if diff > 1.5*np.pi:
            ## jump from -pi to pi
            offset = -2*np.pi
        elif diff < 1.5*np.pi:
            offset = 2*np.pi
        else:
            pass
        new_list.append( pp[1]+offset)
    return np.array(new_list)

def poly_fit_phi(X, Y):
    """X is z, Y is phi"""
    # phi is constrained to [-pi, pi]
    order = 1
    Y = correct_phi(Y)
    pp = np.polyfit(X, Y, order)
    f_y = np.polyval(pp, X)
    diff = np.sum(np.sqrt((f_y - Y)**2/Y**2))
    return pp, f_y, diff


def get_track_parameters(x, y, z):
    # find the center of helix in x-y plane
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def fnc(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    r3 = np.sqrt(x**2 + y**2 + z**2)
    p_zr0 = np.polyfit(r3, z, 1, full=True)
    # res0 = p_zr0[1][0]/x.shape[0]
    p_zr = p_zr0[0]

    theta = np.arccos(p_zr[0])
    # theta = np.arccos(z[0]/r3[0])
    eta = -np.log(np.tan(theta/2.))

    center_estimate = np.mean(x), np.mean(y)
    trans_center, ier = optimize.leastsq(fnc, center_estimate)
    x0, y0 = trans_center
    R = calc_R(*trans_center).mean()

    # d0, z0
    d0 = abs(np.sqrt(x0**2 + y0**2) - R)

    r = np.sqrt(x**2 + y**2)
    p_rz = np.polyfit(r, z, 1)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(d0)


    def quadratic_formular(a, b, c):
        if a == 0:
            return (-c/b, )
        x1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        x2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        return (x1, x2)

    # find the closest approaching point in x-y plane
    int_a = 1 + y0**2/x0**2
    int_b = -2*(x0 + y0**2/x0)
    int_c = x0**2 + y0**2 - R**2
    int_x0, int_x1 = quadratic_formular(int_a, int_b, int_c)
    x1 = int_x0 if abs(int_x0) < abs(int_x1) else int_x1
    y1 = y0*x1/x0
    phi = np.arctan2(y1, x1)

    # track travels colockwise or anti-colockwise
    # positive for colckwise
    xs = x[0] if x[0] != 0 else 1e-1
    ys = y[0] if y[0] != 0 else 1e-1
    is_14 = xs > 0
    is_above = y0 > ys/xs*x0
    sgn = 1 if is_14^is_above else -1

    # last entry is pT*(charge sign)
    return (d0, z0, phi, eta, 0.6*sgn*R/1000)