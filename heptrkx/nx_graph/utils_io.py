"""Handle read and write objects"""

import numpy as np
import pandas as pd
import os

from graph_nets import utils_np
from .prepare import graph_to_input_target

ckpt_name = 'checkpoint_{:05d}.ckpt'

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


def load_data_dicts(file_name):
    with np.load(file_name) as f:
        return dict(f.items())

def load_input_target_data_dicts(path, evtid, isec):
    base_name = os.path.join(path, 'event{:09d}_g{:09d}_{}.npz')

    input_dd = load_data_dicts(base_name.format(evtid, isec, "INPUT"))
    target_dd = load_data_dicts(base_name.format(evtid, isec, "TARGET"))
    return input_dd, target_dd


INPUT_NAME = "INPUT"
TARGET_NAME = "TARGET"
def save_networkx(graph, output_name):
    output_data_name = output_name+'_{}.npz'.format(INPUT_NAME)
    if os.path.exists(output_data_name):
        print(output_data_name, "is there")
        return

    dirname = os.path.dirname(output_data_name)
    os.makedirs(dirname, exist_ok=True)

    input_graph, target_graph = graph_to_input_target(graph, no_edge_feature=True)
    output_data = utils_np.networkx_to_data_dict(input_graph)
    target_data = utils_np.networkx_to_data_dict(target_graph)

    np.savez( output_data_name, **output_data)
    np.savez( output_data_name.replace(INPUT_NAME, TARGET_NAME), **target_data)
