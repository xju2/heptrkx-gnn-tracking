#!/usr/bin/env python
import os

from nx_graph.prepare import hitsgraph_to_networkx_graph
from nx_graph.prepare import graph_to_input_target

from datasets.graph import load_graph

file_format = 'event00000{}_g{:03d}_{}.npz'
INPUT_NAME = "INPUT"
TARGET_NAME = "TARGET"

def get_saver(input_dir_, output_dir_):
    input_dir = input_dir_
    output_dir = output_dir_
    def save_hitsgraph(evt_id, isec):
        input_name = os.path.join(
            input_dir,
            'event00000{}_g{:03d}.npz'.format(evt_id, isec))
        graph = hitsgraph_to_networkx_graph(load_graph(input_name))
        input_graph, target_graph = graph_to_input_target(graph)

        input_data = utils_np.networkx_to_data_dict(input_graph)
        target_data = utils_np.networkx_to_data_dict(target_graph)

        np.savez(
            os.path.join(output_dir,
                         'event00000{}_g{:03d}_{}.npz'.format(evt_id, isec, INPUT_NAME)),
            **input_data)
        np.savez(
            os.path.join(output_dir,
                         'event00000{}_g{:03d}_{}.npz'.format(evt_id, isec, TARGET_NAME)),
            **target_data)

    return save_hitsgraph

def get_loader(input_dir_):
    """
    Load data dict and convert to graph tuples that could
    be directly feeded to TF
    """
    input_dir = input_dir_
    def load_nxgraph(evt_id, isec):
        input_name = os.path.join(
            input_dir,
            'event00000{}_g{:03d}_{}.npz'.format(evt_id, isec, INPUT_NAME))
        # load input
        with np.load(input_name) as f:
            input_data_dict = dict(f.items())
        input_G = utils_np.data_dicts_to_graphs_tuple([input_data_dict])

        target_name= os.path.join(
            input_dir,
            'event00000{}_g{:03d}_{}.npz'.format(evt_id, isec, TARGET_NAME))
        with np.load(file_name.format(TARGET_NAME)) as f:
            target_data_dict = dict(f.items())
        target_G = utils_np.data_dicts_to_graphs_tuple([target_data_dict])

        return input_G, target_G

    return load_nxgraph

if __name__ == "__main__":
    import numpy as np
    import glob
    import re

    from graph_nets import utils_np

    from nx_graph.utils_train import load_config

    import argparse

    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('config',  nargs='?', default='configs/nxgraph_default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = config['data']['input_hitsgraph_dir']
    base_dir = os.path.join(input_dir,'event00000{}_g{:03d}.npz')

    file_patten = base_dir.format(1000, 0).replace('1000', '*')
    all_files = glob.glob(file_patten)
    n_events = len(all_files)
    evt_ids = [int(re.search('event00000([0-9]*)_g000.npz', os.path.basename(x)).group(1))
               for x in all_files]

    section_patten = base_dir.format(1000, 0).replace('_g000', '_g[0-9]*[0-9]*[0-9]')
    n_sections = len(glob.glob(section_patten))
    n_total = n_events*n_sections
    print("Total Events: {} with {} sections, total {} files ".format(
        n_events, n_sections, n_total))

    output = config['data']['output_nxgraph_dir']
    if not os.path.exists(output):
        os.makedirs(output)

    saver = get_saver(input_dir, output)

    print("Input directory: {}".format(input_dir))
    print("Output directory: {}".format(output))

    import time
    start_time = time.time()

    log_name = os.path.join(output, config['data']['log_name'])

    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), # (event number), T (elapsed seconds)\n"
    with open(log_name, 'a') as f:
        f.write(out_str)

    out_str = ""
    for ii, evt_id in enumerate(sorted(evt_ids)):
        for isec in range(n_sections):
            saver(evt_id, isec)

        elapsed_time = time.time() - start_time
        info = "# {:05d}, # {:05d}, T {:.1f}\n".format(ii, evt_id, elapsed_time)
        out_str += info
        print(info)

    with open(log_name, 'a') as f:
        f.write(out_str)
