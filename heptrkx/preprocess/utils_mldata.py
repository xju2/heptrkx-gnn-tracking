"""
process in Tracking ML data
"""
from trackml.dataset import load_event

import pandas as pd
import numpy as np

from scipy import optimize

import os


import yaml
def read_event(evtid, config, info=False):
    with open(config) as f:
        config = yaml.load(f)

    data_dir = config['track_ml']['dir']
    return read(data_dir, evtid, info)


def reconstructable_pids(particles, truth):
    truth_particles = particles.merge(truth, on='particle_id', how='left')
    reconstructable_particles = truth_particles[~np.isnan(truth_particles.weight)]
    return np.unique(reconstructable_particles.particle_id)


def create_segments(hits, layer_pairs, gid_keys='layer',
                    only_true=False):
    hit_gid_groups = hits.groupby(gid_keys)

    def calc_dphi(phi1, phi2):
        """Computes phi2-phi1 given in range [-pi,pi]"""
        dphi = phi2 - phi1
        dphi[dphi > np.pi] -= 2*np.pi
        dphi[dphi < -np.pi] += 2*np.pi
        return dphi

    def cal_deta(hitpair):
        r1 = hitpair.r_out
        r2 = hitpair.r_in
        z1 = hitpair.z_out
        z2 = hitpair.z_in

        R1 = np.sqrt(r1**2 + z1**2)
        R2 = np.sqrt(r2**2 + z2**2)
        theta1 = np.arccos(z1/R1)
        theta2 = np.arccos(z2/R2)
        eta1 = -np.log(np.tan(theta1/2.0))
        eta2 = -np.log(np.tan(theta2/2.0))
        return eta1 - eta2

    # Loop over geometry ID pairs
    for gid1, gid2 in layer_pairs:
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)

        if only_true:
            # much faster operation
            hits1 = hits1[hits1.particle_id != 0]
            hits2 = hits2[hits2.particle_id != 0]
            hit_pairs = pd.merge(
                hits1.reset_index(), hits2.reset_index(),
                how='inner', on='particle_id', suffixes=('_in', '_out'))
            y = 1.0
        else:
            # Join all hit pairs together
            hit_pairs = pd.merge(
                hits1.reset_index(), hits2.reset_index(),
                how='inner', on='evtid', suffixes=('_in', '_out'))
            # Identify the true pairs
            y = (hit_pairs.particle_id_in == hit_pairs.particle_id_out) & (hit_pairs.particle_id_in != 0)

        # Calculate coordinate differences
        dphi = calc_dphi(hit_pairs.phi_in, hit_pairs.phi_out)
        dz = hit_pairs.z_out - hit_pairs.z_in
        dr = hit_pairs.r_out - hit_pairs.r_in
        phi_slope = dphi / dr
        z0 = hit_pairs.z_in - hit_pairs.r_in * dz / dr
        deta = cal_deta(hit_pairs)

        # slopeRZ = np.arctan2(dr, dz)

        selected_features = ['evtid', 'index_in', 'index_out',
                             'hit_id_in', 'hit_id_out',
                             'x_in', 'x_out', 'y_in', 'y_out', 'z_in', 'z_out',
                             'layer_in', 'layer_out']
        if 'lx_in' in hit_pairs.columns:
            selected_features += ['lx_in', 'lx_out', 'ly_in', 'ly_out', 'lz_in', 'lz_out']

        try:
            deta1 = hit_pairs.geta_out - hit_pairs.geta_in
            dphi1 = hit_pairs.gphi_out - hit_pairs.gphi_in
        except [KeyError, AttributeError]:
            deta1 = None
            dphi1 = None

        # Put the results in a new dataframe
        hit_pairs = hit_pairs[selected_features].assign(
            dphi=dphi, dz=dz, dr=dr, true=y, phi_slope=phi_slope, z0=z0, deta=deta)
        if deta1 is not None:
            hit_pairs = hit_pairs.assign(deta1=deta1, dphi1=dphi1)

        hit_pairs = hit_pairs.rename(columns={'index_in': 'hit_idx_in', "index_out": 'hit_idx_out'})

        yield hit_pairs