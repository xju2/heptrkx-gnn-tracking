"""
Some math functions
"""
import numpy as np
import math

def cartesion_to_spherical(x, y, z):
    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r3)
    return r3, theta, phi

def theta_to_eta(theta):
    return -np.log(np.tan(0.5*theta))

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


def ratio_error(a, b, in_percentage=False):
    ratio = a/b
    if in_percentage:
        ratio *= 100
    error = ratio * math.sqrt((a+b)/(a*b))
    return ratio, error

def cov_r(x, y, xe, ye):
    r = math.sqrt(x**2 + y**2)
    dr = math.sqrt(x**2/r**2 * xe**2 + y**2/r**2 * ye**2)
    return r, dr