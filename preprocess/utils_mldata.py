"""
process in Tracking ML data
"""
from trackml.dataset import load_event

from postprocess import utils_fit

import pandas as pd
import numpy as np

from scipy import optimize

import os

def read(data_dir, black_dir, evtid):
    prefix = os.path.join(data_dir, 'event{:09d}'.format(evtid))
    prefix_bl = os.path.join(black_dir, 'event{:09d}-blacklist_'.format(evtid))

    hits_exclude = pd.read_csv(prefix_bl+'hits.csv')
    particles_exclude = pd.read_csv(prefix_bl+'particles.csv')

    hits, particles, truth, cells = load_event(prefix, parts=['hits', 'particles', 'truth', 'cells'])
    hits = hits[~hits['hit_id'].isin(hits_exclude['hit_id'])]
    particles = particles[~particles['particle_id'].isin(particles_exclude['particle_id'])]

    px = particles.px
    py = particles.py
    pt = np.sqrt(px**2 + py**2)
    particles = particles.assign(pt=pt)

    return hits, particles, truth, cells


def reconstructable_pids(particles, truth):
    truth_particles = particles.merge(truth, on='particle_id', how='left')
    reconstructable_particles = truth_particles[~np.isnan(truth_particles.weight)]
    return np.unique(reconstructable_particles.particle_id)


def create_segments(hits, layers, gid_keys='layer'):
    segments = []
    hit_gid_groups = hits.groupby(gid_keys)


    # Loop over geometry ID pairs
    for gid1, gid2 in utils_fit.pairwise(layers):
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)

        # Join all hit pairs together
        hit_pairs = pd.merge(
            hits1.reset_index(), hits2.reset_index(),
            how='inner', on='evtid', suffixes=('_in', '_out'))

        # Calculate coordinate differences
        dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
        deta = cal_deta(hit_pairs)

        # Identify the true pairs
        y = (hit_pairs.particle_id_1 == hit_pairs.particle_id_2) & (hit_pairs.particle_id_1 != 0)

        # Put the results in a new dataframe
        df_pairs = hit_pairs[['evtid', 'index_in', 'index_out', 'hit_id_in', 'hit_id_out', 'layer_in', 'layer_out']].assign(dphi=dphi, dz=dz, dr=dr, true=y, phi_slope=phi_slope, z0=z0, deta=deta)

        print('processed:', gid1, gid2, "True edges {} and Fake Edges {}".format(df_pairs[df_pairs['y']==True].shape[0], df_pairs[df_pairs['y']==False].shape[0]))

        df_pairs = df_pairs.rename(columns={'index_in': 'hit_idx_in', "index_out": 'hit_idx_out'})

        segments.append(df_pairs)

    merged_segments = pd.concat(segments, ignore_index=True)
    return merged_segments


def get_track_parameters(x, y, z):
    # find the center of helix in x-y plane
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def fnc(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    r3 = np.sqrt(x**2 + y**2 + z**2)
    p_zr0 = np.polyfit(r3, z, 1, full=True)
    res0 = p_zr0[1][0]/x.shape[0]
    p_zr = p_zr0[0]

#    if res0 > 10:
#        p_zr1 = np.polyfit(r3[:-1], z[:-1], 1, full=True)
#        res1 = p_zr1[1][0]/(x.shape[0] - 1)
#        if res1 < res0:
#            print("Drop Last", res1, res0)
#            r3 = r3[:-1]
#            x = x[:-1]
#            y = y[:-1]
#            z = z[:-1]
#            p_zr = p_zr1[0]

    #theta = np.arccos(p_zr[0])
    theta = np.arccos(z[0]/r3[0])
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
    xs = x[0]
    ys = y[0]
    is_14 = xs > 0
    is_above = y0 > ys/xs*x0
    sgn = 1 if is_14^is_above else -1

    # last entry is pT*(charge sign)
    return (d0, z0, phi, eta, 0.6*sgn*R/1000)


def local_angle(cell, module):
    n_u = max(cell['ch0']) - min(cell['ch0'])
    n_v = max(cell['ch1']) - min(cell['ch1'])
    l_u = n_u * module.pitch_u.values   # x
    l_v = n_v * module.pitch_v.values   # y
    l_w = 2   * module.module_t.values  # z
    return (l_u, l_v, l_w)


def module_info(detector_dir):
    detector = pd.read_csv(detector_dir)

    def get_fnc(volume_id, layer_id, module_id):
        return detector[ (detector.volume_id == volume_id) & (detector.layer_id == layer_id) & (detector.module_id == module_id) ]
    return get_fnc


def extract_rotation_matrix(module):
    rot_matrix = np.matrix( [[ module.rot_xu.values[0], module.rot_xv.values[0], module.rot_xw.values[0]],
                            [  module.rot_yu.values[0], module.rot_yv.values[0], module.rot_yw.values[0]],
                            [  module.rot_zu.values[0], module.rot_zv.values[0], module.rot_zw.values[0]]])
    return rot_matrix, np.linalg.inv(rot_matrix)
