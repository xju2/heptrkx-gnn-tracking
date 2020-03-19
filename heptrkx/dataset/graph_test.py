#!/usr/bin/env python
"""
Test make_graph_ntuples
"""
from heptrkx.dataset.graph import make_graph_ntuples
from heptrkx.dataset.graph import DoubletGraphGenerator
import pandas as pd


hit_file_name = '/global/cscratch1/sd/xju/heptrkx/codalab/inputs/hitfiles/evt21001_test.h5'
doublet_file_name = '/global/cscratch1/sd/xju/heptrkx/codalab/inputs/doublet_files/doublets-evt21001_test.h5'

def test_graph():
    with pd.HDFStore(hit_file_name, 'r') as hit_store:
        print(hit_store.keys())
        with pd.HDFStore(doublet_file_name, 'r') as doublet_store:
            print(doublet_store.keys())
            key = 'evt21001'
            hit = hit_store[key]
            doublets = []
            for ipair in range(9):
                pair_key = key+'/pair{}'.format(ipair)
                doublets.append(doublet_store[pair_key])
            doublets = pd.concat(doublets)
            print("{:,} hits and {:,} doublets".format(hit.shape[0], doublets.shape[0]))

            all_graphs = make_graph_ntuples(
                                hit, doublets, 2, 10,
                                verbose=True)

    return all_graphs


def test_dataset():
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'])
    graph_gen.add_file(hit_file_name, doublet_file_name)
    dataset = graph_gen.create_training_dataset()


if __name__ == "__main__":
    # test_graph()
    test_dataset()