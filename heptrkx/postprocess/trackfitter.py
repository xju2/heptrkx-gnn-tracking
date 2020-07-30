"""
Track Fitting
"""

from scipy import optimize
import numpy as np
import math

def helix_fitter(x, y, z):
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
    return (d0, z0, eta, phi, 0.6*sgn*R/1000)    


def conformal_mapping(x, y, z):
    """
    x, y, z: np.array([])
    return: 
    """
    # ref. 10.1016/0168-9002(88)90722-X
    r = x**2 + y**2
    u = x/r
    v = y/r
    # assuming the imapact parameter is small
    # the v = 1/(2b) - u x a/b - u^2 x epsilon x (R/b)^3
    pp, vv = np.polyfit(u, v, 2, cov=True)
    b = 0.5/pp[2]
    a = -pp[1]*b
    R = math.sqrt(a**2 + b**2)
    e = -pp[0] / (R/b)**3 # approximately equals to d0
    magnetic_field = 2.0
    pT = 0.3*magnetic_field*R/1000 # in GeV
    # print(a, b, R, e, pT)

    p_rz = np.polyfit(np.sqrt(r), z, 2)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(abs(e))

    r3 = np.sqrt(r + z**2)
    p_zr = np.polyfit(r3, z, 2)
    cos_val = p_zr[0]*z0 + p_zr[1]
    theta = np.arccos(cos_val)
    eta = -np.log(np.tan(theta/2.))
    phi = math.atan2(b, a)

    return e, z0, eta, phi, pT