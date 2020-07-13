"""
make doublets
"""

import os
import pandas as pd

from heptrkx.utils_math import calc_dphi


def create_segments(hits, layer_pairs,
                    only_true=True, verbose=False):
    """
    create true segements, assuming hits has truth info
    """
    gid_keys = 'layer'
    hit_gid_groups = hits.groupby(gid_keys)
    
    # segments = []
    for gid1, gid2 in layer_pairs:
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)

        if only_true:
            hit_pairs = pd.merge(
                hits1.reset_index(), hits2.reset_index(),
                how='inner', on='particle_id', suffixes=('_in', '_out'))
            hit_pairs = hit_pairs.assign(solution=True)
        else:
            hit_pairs = pd.merge(
                hits1.reset_index(), hits2.reset_index(),
                how='inner', on='evtid', suffixes=('_in', '_out'))
            # Identify the true pairs
            solution = (hit_pairs.particle_id_in == hit_pairs.particle_id_out) \
                & (hit_pairs.particle_id_in != 0)
            hit_pairs = hit_pairs.assign(solution=solution)

        if verbose:
            print("{}-{} has {:,} doublets".format(gid1, gid2, hit_pairs.shape[0]))

        yield hit_pairs
        

def calculate_segment_features(segments):
    dr = segments.r_out - segments.r_in
    dz = segments.z_out - segments.z_in
    zorg = segments.z_out - segments.r_out * dz / dr
    dphi = calc_dphi(segments.phi_in, segments.phi_out)
    phi_slope = dphi/dr

    deta = segments.eta_out - segments.eta_in
    deta1 = segments.geta_out - segments.geta_in
    dphi1 = calc_dphi(segments.gphi_out, segments.gphi_in)


    return {
        'dphi': dphi,
        'dz': dz,
        'dr': dr,
        "phi_slope": phi_slope,
        "z0":zorg,
        'deta': deta,
        'deta1': deta1,
        'dphi1': dphi1,
    }