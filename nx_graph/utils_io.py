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
