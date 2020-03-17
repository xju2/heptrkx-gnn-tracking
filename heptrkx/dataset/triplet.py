"""
Check Triplet performance
"""
import glob
import pandas as pd
import numpy as np
import pickle

def read_triplets(seed_candidates):
    """
    Read the input seed candidates
    """
    if "pickle" in seed_candidates:
        if "*" in seed_candidates:
            all_files = glob.glob(seed_candidates)
            new_data = []
            for file_name in all_files:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
                    for dd in data:
                        new_data.append((dd[0], dd[1], dd[2], dd[3]))
            df_seed = pd.DataFrame(new_data, columns=['evtid', 'h1', 'h2', 'h3'], dtype=np.int64)
        else:
            with open(seed_candidates, 'rb') as f:
                data = pickle.load(f)
                new_data = []
                for dd in data:
                    new_data.append((dd[0], dd[1], dd[2], dd[3]))
                    # idx = int(dd[0][10:])
                    # new_data.append((idx, dd[1], dd[2], dd[3]))
                df_seed = pd.DataFrame(new_data, columns=['evtid', 'h1', 'h2', 'h3'], dtype=np.int64)
    else:
        column_names = ['evtid', 'h1', 'h2', 'h3']
        if "*" in seed_candidates:
            all_files = glob.glob(seed_candidates)
            new_data = []
            for file_name in all_files:
                df_seed_tmp = pd.read_csv(file_name, header=None, names=column_names,)
                new_data.append(df_seed_tmp)
            df_seed = pd.concat(new_data)
        else:
            df_seed = pd.read_csv(seed_candidates, header=None,
                                names=column_names)
    return df_seed


def evaluate_evt(event, seed_candidates, min_hits=5, layers=None, verbose=False):
    hits = event.hits
    evtid = hits.evtid.values[0]
    all_particles = np.unique(hits.particle_id).shape[0]
    all_hits = hits.shape[0]
    if verbose:
        print("Total particles: {}, with {} hits".format(all_particles, all_hits))

    aa = hits.groupby(['particle_id'])['hit_id'].count()
    total_particles = aa[aa > min_hits].index
    total_particles = total_particles[total_particles != 0]
    n_total_particles = total_particles.shape[0]
    if verbose:
        print("Event {} has {} particles with minimum of {} hits".format(
            evtid, n_total_particles, min_hits))

    df = seed_candidates
    if verbose:
        print("Event {} has {} seed candidates".format(evtid, df.shape[0]))

    if layers is not None:
        ## now select the hits in specified layers
        hits = hits[hits.layer.isin(layers)]

    # particles leaving 3 hits in three layers
    bb = hits.groupby(['particle_id'])['layer'].count()
    good_particles = bb[(bb > 2) & (aa > min_hits)].index
    good_particles = good_particles[good_particles != 0]

    if verbose:
        print("Event {} has {} particles leaving hits at inner 3 layers".format(
            evtid, good_particles.shape[0]))
        good_particles_pT = np.unique(hits[hits.particle_id.isin(good_particles) \
                                & hits.pt.abs() >= 1].particle_id)
        print("Event {} has {} particles leaving hits at inner 3 layers, with pT > 1 GeV".format(
            evtid, good_particles_pT.shape[0]))

    df1 = df.merge(hits, left_on='h1', right_on='hit_id', how='left')
    df2 = df.merge(hits, left_on='h2', right_on='hit_id', how='left')
    df3 = df.merge(hits, left_on='h3', right_on='hit_id', how='left')
    p1 = df1.particle_id.values.astype('int64')
    p2 = df2.particle_id.values.astype('int64')
    p3 = df3.particle_id.values.astype('int64')

    n_total_seeds = df.shape[0]
    true_seeds_dup = p1[(p1 != 0) & (p1==p2) & (p2==p3)]
    n_true_seeds_dup = true_seeds_dup.shape[0]
    true_seeds = p1[(p1 != 0) & (p1==p2) & (p2==p3) \
        & (df1.layer != df2.layer)\
        & (df1.layer != df3.layer) & (df2.layer != df3.layer)]
    n_true_seeds = true_seeds.shape[0]

    # unique true seeds should be  part of good particles
    dup_mask = np.isin(true_seeds, good_particles)
    unique_true_seeds = np.unique(true_seeds[dup_mask])
    n_unique_true_seeds = unique_true_seeds.shape[0]

    if verbose:
        print("{} particles matched".format(n_unique_true_seeds))
        print("Fraction of duplicated seeds: {:.2f}%".format(100 - n_unique_true_seeds*100/n_true_seeds))
        print("Purity: {:.2f}%".format(n_true_seeds*100./n_total_seeds))
        print("Efficiency: {:.2f}%".format(n_unique_true_seeds*100./good_particles.shape[0]))
    
    summary = {
        "evtid": evtid,
        'n_hits': hits.shape[0],
        'n_particles': good_particles.shape[0],
        'n_matched_particles': unique_true_seeds.shape[0],
        'n_seeds': n_total_seeds,
        'n_true_seeds_dup': true_seeds_dup.shape[0],
        'n_true_seeds': true_seeds.shape[0]
    }
    
    df_unique_true_seeds = pd.DataFrame(unique_true_seeds, columns=['particle_id'])
    df_unique_true_seeds = df_unique_true_seeds.merge(event.particles, on='particle_id', how='left')
    df_total_particles = pd.DataFrame(good_particles, columns=['particle_id'])
    df_total_particles = df_total_particles.merge(event.particles, on='particle_id', how='left')

    return (summary, df_unique_true_seeds, df_total_particles)

