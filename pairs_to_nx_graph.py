#!/usr/bin/env python3
import numpy as np
import pandas as pd

from trackml.dataset import load_event
import networkx as nx

import os
import re

from nx_graph.prepare import get_edge_features
from nx_graph.prepare import graph_to_input_target
from graph_nets import utils_np

INPUT_NAME = 'INPUT'
TARGET_NAME = "TARGET"

vlids = [(7,2), (7,4), (7,6), (7,8), (7,10), (7,12), (7,14),
         (8,2), (8,4), (8,6), (8,8),
         (9,2), (9,4), (9,6), (9,8), (9,10), (9,12), (9,14),
         (12,2), (12,4), (12,6), (12,8), (12,10), (12,12),
         (13,2), (13,4), (13,6), (13,8),
         (14,2), (14,4), (14,6), (14,8), (14,10), (14,12),
         (16,2), (16,4), (16,6), (16,8), (16,10), (16,12),
         (17,2), (17,4),
         (18,2), (18,4), (18,6), (18,8), (18,10), (18,12)]
n_det_layers = len(vlids)


def read_pairs_input(file_name):
    pairs = []
    iline = 0
    with open(file_name) as f:
        for line in f:
            if iline == 0:
                n_pairs = int(line[:-1])
                iline += 1
                continue
            pairs.append([int(x) for x in line[:-1].split()])
    return pairs


def pairs_to_graph(pairs, evt_file_name, output_dir):
    n_pairs_per_file = 20000

    evt_id = int(re.search('event00000([0-9]*)', os.path.basename(evt_file_name)).group(1))

    hits_org, particles, truth = load_event(
        evt_file_name, parts=['hits', 'particles', 'truth'])

    n_particles = particles.shape[0]
    truth = truth.merge(particles[['particle_id']], on='particle_id')
    hits_org2 = hits_org.merge(truth[['hit_id', 'particle_id']], on='hit_id', how='left')
    hits_org2 = hits_org2.fillna(value=0)

    # Assign convenient layer number [0-49]
    vlid_groups = hits_org2.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])

    # add new features
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    hits = hits.assign(r=r, phi=phi)

    # add hit indexes to column hit_idx
    hits_with_idx = hits.rename_axis('hit_idx').reset_index()
    feature_scale = np.array([1000., np.pi, 1000.])
    print(hits_with_idx.columns)

    # get features and determine if it is a real edge
    total_pairs = len(pairs)
    n_files = int(total_pairs/n_pairs_per_file) + 1
    print("event", evt_id, "has", n_files, "files")

    # dealing with pairs
    pairs_arr = np.array(pairs)
    df_in_nodes  = pd.DataFrame(pairs_arr[:, 0], columns=['hit_id'])
    df_out_nodes = pd.DataFrame(pairs_arr[:, 1], columns=['hit_id'])

    df_in_nodes  = df_in_nodes.merge(hits_with_idx, on='hit_id', how='left')
    df_out_nodes = df_out_nodes.merge(hits_with_idx, on='hit_id', how='left')

    # find out if edge is true edge from particle ID != 0
    n_edges = df_in_nodes.shape[0]
    y = np.zeros(n_edges, dtype=np.float32)
    pid1 = df_in_nodes ['particle_id'].values
    pid2 = df_out_nodes['particle_id'].values
    y[:] = (pid1 == pid2) & (pid1 != 0)


    # build graph
    for ifile in range(n_files):
        output_name = os.path.join(output_dir, 'event{:09d}_g{:03d}_{}.npz')

        # while hits will be used in the graph
        used_hits_set = set()
        for i in range(n_pairs_per_file):
            idx = ifile*n_pairs_per_file + i
            if idx >= total_pairs-1:
                break

            in_node_idx  = int(df_in_nodes.iloc[idx, 1]) # 1 is hit_idx
            out_node_idx = int(df_out_nodes.iloc[idx, 1])
            used_hits_set.add(in_node_idx)
            used_hits_set.add(out_node_idx)

        # a dictionary to keep track of hit_idx and node_idx
        graph = nx.DiGraph()
        hits_id_dict = {}
        for ii,idx in enumerate(used_hits_set):
            hits_id_dict[idx] = ii
            graph.add_node(ii, pos=hits_org.iloc[idx][['r', 'phi', 'z']].values/feature_scale, solution=0.0)


        for i in range(n_pairs_per_file):
            idx = ifile*n_pairs_per_file + i
            if idx >= total_pairs-1:
                break

            in_hit_idx  = int(df_in_nodes.iloc[idx, 1])
            out_hit_idx = int(df_out_nodes.iloc[idx, 1])

            in_node_idx = hits_id_dict[in_hit_idx]
            out_node_idx = hits_id_dict[out_hit_idx]
            f1 = graph.node[in_node_idx]['pos']
            f2 = graph.node[out_node_idx]['pos']
            distance = get_edge_features(f1, f2)
            graph.add_edge(in_node_idx,  out_node_idx, distance=distance, solution=y[idx])
            graph.add_edge(out_node_idx, in_node_idx,  distance=distance, solution=y[idx])
            graph.node[in_node_idx].update(solution=y[idx])
            graph.node[out_node_idx].update(solution=y[idx])


        graph.graph['features'] = np.array([0.])
        input_graph, target_graph = graph_to_input_target(graph)
        input_data = utils_np.networkx_to_data_dict(input_graph)
        target_data = utils_np.networkx_to_data_dict(target_graph)
        input_data_name = os.path.join(
            output_dir,
            output_name.format(evt_id, ifile, INPUT_NAME))
        np.savez( input_data_name, **input_data)
        np.savez( input_data_name.replace(INPUT_NAME, TARGET_NAME), **target_data)


if __name__ == "__main__":
    pairs_input_dir = '/global/homes/x/xju/track/gnn/code/top-quarks/trained/output_pairs'
    output_dir = '/global/cscratch1/sd/xju/heptrkx/data/graph_from_pairs_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    import time
    start_time = time.time()

    evt_ids = [1000, 1002, 1003, 1004]

    log_name = os.path.join(output_dir, "timing.log")
    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), # (event number), T (elapsed seconds)\n"
    with open(log_name, 'a') as f:
        f.write(out_str)

    out_str = ""
    for ii,evt_id in enumerate(evt_ids):
        pairs = read_pairs_input(os.path.join(pairs_input_dir, 'pairs_{}'.format(evt_id)))
        evt_file_name = '/global/cscratch1/sd/xju/heptrkx/trackml_inputs/train_all/event{:09d}'.format(evt_id)
        pairs_to_graph(pairs, evt_file_name, output_dir)
        elapsed_time = time.time() - start_time
        info = "# {:05d}, # {:05d}, T {:.1f}\n".format(ii, evt_id, elapsed_time)
        out_str += info
        print(info)

    with open(log_name, 'a') as f:
        f.write(out_str)
