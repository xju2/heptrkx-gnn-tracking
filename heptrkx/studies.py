"""Functions that are used by scripts
"""
import numpy as np

from heptrkx import load_yaml, select_pair_layers, layer_pairs
from heptrkx.preprocess import utils_mldata
from heptrkx.nx_graph import utils_data
from heptrkx import seeding

def fraction_of_duplicated_hits(evtid, config_name):
    config = load_yaml(config_name)
    evt_dir = config['track_ml']['dir']

    hits, particles, truth, cells = utils_mldata.read(evt_dir, evtid)
    hits = utils_data.merge_truth_info_to_hits(hits, particles, truth)
    layers = config['doublets_from_cuts']['layers']
    barrel_hits = hits[hits.layer.isin(layers)].assign(evtid=evtid)
    sel_layer_id = select_pair_layers(layers)

    # remove noise hits
    barrel_hits = barrel_hits[barrel_hits.particle_id > 0]


    sel = barrel_hits.groupby("particle_id")['layer'].apply(
        lambda x: len(x) - np.unique(x).shape[0]
    ).values
    return sel


def eff_purity_of_edge_selection(evtid, config_name):
    config = load_yaml(config_name)
    evt_dir = config['track_ml']['dir']

    hits, particles, truth, cells = utils_mldata.read(evt_dir, evtid)
    hits = utils_data.merge_truth_info_to_hits(hits, particles, truth)
    layers = config['doublets_from_cuts']['layers']
    sel_layer_id = select_pair_layers(layers)
    barrel_hits = hits[hits.layer.isin(layers)].assign(evtid=evtid)

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


