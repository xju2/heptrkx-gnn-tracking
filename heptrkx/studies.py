"""Functions that are used by scripts
"""
import os

import numpy as np
import pandas as pd

from heptrkx import load_yaml, select_pair_layers, layer_pairs
from heptrkx.nx_graph import utils_data
from heptrkx import seeding
from heptrkx.master import Event

from heptrkx.postprocess import wrangler, analysis

def fraction_of_duplicated_hits(evtid, config_name):
    config = load_yaml(config_name)
    evt_dir = config['track_ml']['dir']
    layers = config['doublets_from_cuts']['layers']

    event = Event(evt_dir, evtid)
    barrel_hits = event.filter_hits(layers)

    # remove noise hits
    barrel_hits = barrel_hits[barrel_hits.particle_id > 0]

    sel = barrel_hits.groupby("particle_id")['layer'].apply(
        lambda x: len(x) - np.unique(x).shape[0]
    ).values
    return sel


def eff_purity_of_edge_selection(evtid, config_name):
    config = load_yaml(config_name)
    evt_dir = config['track_ml']['dir']
    layers = config['doublets_from_cuts']['layers']
    sel_layer_id = select_pair_layers(layers)

    event = Event(evt_dir, evtid)
    barrel_hits = event.filter_hits(layers)

    phi_slope_max = config['doublets_from_cuts']['phi_slope_max']
    z0_max = config['doublets_from_cuts']['z0_max']

    tot_list = []
    sel_true_list = []
    sel_list = []
    for pair_idx in sel_layer_id:
        pairs = layer_pairs[pair_idx]
        df = seeding.create_segments(barrel_hits, pairs)
        tot = df[df.true].pt.to_numpy()
        sel_true = df[
            (df.true)\
            & (df.phi_slope.abs() < phi_slope_max)\
            & (df.z0.abs() < z0_max)
        ].pt.to_numpy()
        sel = df[
            (df.phi_slope.abs() < phi_slope_max)\
            & (df.z0.abs() < z0_max)
        ].pt.to_numpy()
        tot_list.append(tot)
        sel_true_list.append(sel_true)
        sel_list.append(sel)

    return (tot_list, sel_true_list, sel_list)

def eff_purity_of_edge_selection2(evtid, evt_dir,
                                  phi_slope_max, z0_max, layers=None, min_hits=0,
                                  verbose=False):

    sel_layer_id = select_pair_layers(layers)

    event = Event(evt_dir, evtid)
    hits = event.filter_hits(layers)
    ## particles having at least mininum number of hits associated
    cut = hits[hits.particle_id != 0].groupby('particle_id')['hit_id'].count() > min_hits
    pids = cut[cut].index
    if verbose:
        print("event {} has {} particles with at least {} hits".format(
            evtid, len(pids), min_hits))

    tot_list = []
    sel_true_list = []
    sel_list = []
    for pair_idx in sel_layer_id:
        layer_pair = layer_pairs[pair_idx]
        df = seeding.create_segments(hits, layer_pair)
        df.loc[~df.particle_id.isin(pids), 'true'] = False
        tot = df[df.true].pt.to_numpy()
        sel_true = df[
            (df.true)\
            & (df.phi_slope.abs() < phi_slope_max)\
            & (df.z0.abs() < z0_max)
        ].pt.to_numpy()
        sel = df[
            (df.phi_slope.abs() < phi_slope_max)\
            & (df.z0.abs() < z0_max)
        ].pt.to_numpy()
        tot_list.append(tot)
        sel_true_list.append(sel_true)
        sel_list.append(sel)
        if verbose:
            print("pair ({}, {}), {} true segments, {} selected, {} true ones selected \
                  segment efficiency {:.2f}% and purity {:.2f}%".format(
                      layer_pair[0], layer_pair[1],
                      tot.shape[0], sel.shape[0], sel_true.shape[0],
                      100.*sel_true.shape[0]/tot.shape[0],
                      100.*sel_true.shape[0]/sel.shape[0]
                  )
                 )

    return (tot_list, sel_true_list, sel_list)


def track_eff_of_edge_selected(evtid, config_name, matching_cut=0.8, remove_duplicated_hits=False):
    config = load_yaml(config_name)
    evt_dir = config['track_ml']['dir']
    layers = config['doublets_from_cuts']['layers']
    sel_layer_id = select_pair_layers(layers)

    event = Event(evt_dir)
    event.read(evtid)
    barrel_hits = event.filter_hits(layers)

    data_source = 'doublets_from_cuts'
    cfg = config[data_source]
    pairs_selected_dir = cfg['selected']
    pairs_input_dir = os.path.join(pairs_selected_dir, 'evt{}'.format(evtid))

    all_segments = []
    for pair_id in sel_layer_id:
        file_name = os.path.join(pairs_input_dir, 'pair{:03d}.h5'.format(pair_id))
        try:
            with pd.HDFStore(file_name, 'r') as store:
                df = store.get('data')
        except KeyError:
            pass
        else:
            all_segments.append(df)

    segments = pd.concat(all_segments, ignore_index=True)
    graph = utils_data.segments_to_nx(
        barrel_hits, segments,
        sender_hitid_name='hit_id_in',
        receiver_hitid_name='hit_id_out',
        solution_name='true',
        use_digraph=True,
        bidirection=False
    )
    track_cands = wrangler.get_tracks(graph, feature_name='solution', with_fit=False)
    df_track_cands = analysis.graphs_to_df(track_cands)
    summary = analysis.summary_on_prediction(graph, barrel_hits, df_track_cands, matching_cut=matching_cut)

    true_nhits = barrel_hits[barrel_hits.particle_id > 0].groupby('particle_id')['hit_id'].count()
    true_particle_ids = true_nhits[true_nhits > 2].index.to_numpy()

    particles = event.particles
    pT_all = particles[particles.particle_id.isin(true_particle_ids)].pt.to_numpy()
    pT_sel = particles[particles.particle_id.isin(summary['correct_pids'])].pt.to_numpy()
    return pT_all, pT_sel
