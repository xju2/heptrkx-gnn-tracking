"""Handle read and write objects"""

import numpy as np
import pandas as pd


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
