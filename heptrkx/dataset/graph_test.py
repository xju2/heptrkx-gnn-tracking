#!/usr/bin/env python
"""
Test make_graph_ntuples
"""
from heptrkx.dataset.graph import make_graph_ntuples
from heptrkx.dataset.graph import DoubletGraphGenerator
from heptrkx.dataset.graph import specs_from_graphs_tuple
import pandas as pd
import tensorflow as tf


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
    training_dataset = graph_gen.create_dataset()
    print(list(training_dataset.take(1).as_numpy_iterator()))

    # testing_dataset = graph_gen.create_dataset()
    # print(list(testing_dataset.take(1).as_numpy_iterator()))

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # dist_training_dataset = mirrored_strategy.experimental_distribute_dataset(training_dataset)
    # with mirrored_strategy.scope():
    #     for inputs in dist_training_dataset:
    #         input, target = inputs
    #         print(input.n_node)
    #         break

def test_signature():
    with_batch_dim = False
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim)
    graph_gen.add_file(hit_file_name, doublet_file_name)
    in_graphs, out_graphs = graph_gen.create_graph(1, is_training=True)

    input_signature = (
        specs_from_graphs_tuple(in_graphs, with_batch_dim),
        specs_from_graphs_tuple(out_graphs, with_batch_dim)
    )
    print("Input graph: ", in_graphs)
    print("Input signature: ", input_signature[0])
    print("Target graph: ", out_graphs)
    print("Target signature:", input_signature[1])

def test_mask():
    with_batch_dim = False
    with_pad = True
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim, with_pad=with_pad)
    graph_gen.add_file(hit_file_name, doublet_file_name)

    input_graph, out_graph = graph_gen.create_graph(1, is_training=True)
    print("Edges:", out_graph.n_edge)

    row_index = tf.range(tf.math.reduce_sum(out_graph.n_edge))
    n_valid_edges = out_graph.n_edge[0]
    mask = tf.cast(row_index < n_valid_edges, tf.float16)
    mask = tf.expand_dims(mask, axis=1)
    print("Mask:", mask)
    print("Mask shape:", tf.shape(mask))
    print("Edge shape", tf.shape(out_graph.edges))
    print("Edge:", out_graph.edges)

    real_weight = 100
    fake_weight = 1.0

    weights = out_graph.edges * real_weight + (1 - out_graph.edges) * fake_weight
    print("Weight shape:", tf.shape(weights))


if __name__ == "__main__":
    # test_graph()
    # test_dataset()
    # test_signature()
    test_mask()