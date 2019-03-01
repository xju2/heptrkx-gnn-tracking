#!/usr/bin/env python3
import numpy as np
import pandas as pd

from trackml.dataset import load_event
import networkx as nx

import os
import re

from nx_graph.prepare import get_networkx_saver
from nx_graph.converters import create_evt_pairs_converter

import multiprocessing as mp

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
    pairs = np.array(pairs)
    return pairs


def save_pairs_to_graphs(pairs, evt_file_name, output_dir, n_workers=1, n_pairs_per_file=30000):

    evt_id = int(re.search('event00000([0-9]*)', os.path.basename(evt_file_name)).group(1))
    total_pairs = pairs.shape[0]
    # get features and determine if it is a real edge
    n_files = int(total_pairs/n_pairs_per_file) + 1
    print("event", evt_id, "splits to", n_files, "files")

    pairs_converter = create_evt_pairs_converter(evt_file_name)
    pair_list = np.array_split(pairs, n_files)

    saver = get_networkx_saver(output_dir)
    ### create workers to each pair
    with mp.Pool(processes=n_workers) as pool:
        graphs = pool.map(pairs_converter, pair_list)

    for ii,graph in enumerate(graphs):
        saver(evt_id, ii, graph)


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Convert pairs to nx-graphs')
    add_arg = parser.add_argument
    add_arg('config',  nargs='?', default='configs/pairs_to_nx.yaml')
    args = parser.parse_args()

    from nx_graph.utils_train import load_config
    config = load_config(args.config)



    pairs_input_dir = config['input_pairs']
    output_dir = config['output_graphs']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    import re
    import glob
    evt_ids = set([int(re.search('pairs_([0-9]*)', os.path.basename(x)).group(1))
               for x in glob.glob(os.path.join(pairs_input_dir, 'pairs_*'))])

    ## check events that are already there
    evt_search_pp = 'event00000([0-9]*)_g000_INPUT.npz'
    evt_ids_ready = set([int(re.search(evt_search_pp, os.path.basename(x)).group(1))
                     for x in glob.glob(os.path.join(output_dir, 'event*_g000_INPUT.npz'))])
    evt_ids = sorted(list(evt_ids.difference(evt_ids_ready)))
    print("events to process:", len(evt_ids))

    import time
    start_time = time.time()

    log_name = os.path.join(output_dir, "timing.log")
    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), # (event number), T (elapsed seconds)\n"
    with open(log_name, 'a') as f:
        f.write(out_str)

    out_str = ""
    evt_file_name = os.path.join(config['input_track_events'], 'event{:09d}')
    for ii,evt_id in enumerate(evt_ids):
        pairs = read_pairs_input(os.path.join(pairs_input_dir, 'pairs_{}'.format(evt_id)))
        evt_name = evt_file_name.format(evt_id)

        save_pairs_to_graphs(
            pairs, evt_name, output_dir,
            configs['n_workers'],
            configs['n_pairs_per_file']
        )

        elapsed_time = time.time() - start_time
        info = "# {:05d}, # {:05d}, T {:.1f}\n".format(ii, evt_id, elapsed_time)
        out_str += info
        print(info)

    with open(log_name, 'a') as f:
        f.write(out_str)
