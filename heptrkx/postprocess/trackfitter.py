"""
Track Fitting
"""

import numpy as np
import math

def helix_fitter(x, y, z):
    pass

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
    magnetic_filed = 2.0
    pT = 0.3*magnetic_filed*R/1000 # in GeV

    p_rz = np.polyfit(np.sqrt(r), z, 1)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(e)

    r3 = np.sqrt(r + z**2)
    p_zr0 = np.polyfit(r3, z, 1)
    p_zr = p_zr0[0]
    theta = np.arccos(p_zr[0])
    eta = -np.log(np.tan(theta/2.))
    phi = math.arccos(b, a) * math.copysign(b)

    return e, z0, eta, phi, pT