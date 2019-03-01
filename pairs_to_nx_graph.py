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


def save_pairs_to_graphs(pairs, evt_file_name, output_dir, n_pairs_per_file=30000):

    evt_id = int(re.search('event00000([0-9]*)', os.path.basename(evt_file_name)).group(1))
    total_pairs = pairs.shape[0]
    # get features and determine if it is a real edge
    n_files = int(total_pairs/n_pairs_per_file) + 1
    print("event", evt_id, "splits to", n_files, "files")

    pairs_converter = create_evt_pairs_converter(evt_file_name)
    pair_list = np.array_split(pairs, n_files)

    saver = get_networkx_saver(output_dir)
    for ii,pair in enumerate(pair_list):
        graph = pairs_converter(pair)
        saver(evt_id, ii, graph)


if __name__ == "__main__":
    from nx_graph.utils_train import load_config

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

        save_pairs_to_graphs(pairs, evt_file_name, output_dir)
        elapsed_time = time.time() - start_time
        info = "# {:05d}, # {:05d}, T {:.1f}\n".format(ii, evt_id, elapsed_time)
        out_str += info
        print(info)

    with open(log_name, 'a') as f:
        f.write(out_str)
