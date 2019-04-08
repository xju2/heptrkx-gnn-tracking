"""
process in Tracking ML data
"""
from trackml.dataset import load_event

from postprocess import utils_fit

import pandas as pd
import numpy as np

import os

def read(data_dir, black_dir, evtid):
    prefix = os.path.join(data_dir, 'event{:09d}'.format(evtid))
    prefix_bl = os.path.join(black_list_dir, 'event{:09d}-blacklist_'.format(evtid))

    hits_exclude = pd.read_csv(prefix_bl+'hits.csv')
    particles_exclude = pd.read_csv(prefix_bl+'particles.csv')

    hits, particles, truth = load_event(prefix, parts=['hits', 'particles', 'truth'])
    hits = hits[~hits['hit_id'].isin(hits_exclude['hit_id'])]
    particles = particles[~particles['particle_id'].isin(particles_exclude['particle_id'])]

    px = particles.px
    py = particles.py
    pt = np.sqrt(px**2 + py**2)
    particles = particles.assign(pt=pt)

    return hits, particles, truth


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
