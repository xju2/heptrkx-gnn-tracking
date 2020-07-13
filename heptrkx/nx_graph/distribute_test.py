#!/usr/bin/env python
"""
Test experimental_make_numpy_dataset
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import graphs

from heptrkx.dataset.graph_test import test_graph
import collections
import numpy as np
import itertools


np.set_printoptions(precision=4)

def gen():
    for i  in itertools.count(1):
        yield (i, [1] * i)

def test_gen():
    dataset = tf.data.Dataset.from_generator(
        gen, (tf.int64, tf.int64),
        (tf.TensorShape([]), tf.TensorShape([None]))
    )
    print(list(dataset.take(3).as_numpy_iterator()))

def gen_dict():
    for i  in itertools.count(1):
        yield {"x": i, "y": i*i}

def test_gen_dict():
    dataset = tf.data.Dataset.from_generator(
        gen_dict,
        {"x": tf.int32, "y": tf.int32},
        {"x": tf.TensorShape([]), 'y': tf.TensorShape([])}
    )
    print(list(dataset.take(3).as_numpy_iterator()))

def dtype_shape_from_graphs_tuple(
    input_graph, 
    dynamic_num_graphs=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    ):
    graphs_tuple_dtype = {}
    graphs_tuple_shape = {}

    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]
    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(input_graph, field_name)
        shape = list(field_sample.shape)
        dtype = field_sample.dtype
        print(field_name, shape, dtype)

        if (shape and (dynamic_num_graphs
                        or (dynamic_num_nodes and field_name == graphs.NODES)
                        or (dynamic_num_edges and field_name in edge_dim_fields)
                    )
        ): shape[0] = None

        graphs_tuple_dtype[field_sample] = dtype
        graphs_tuple_shape[field_name] = shape
    
    return graphs_tuple_dtype, graphs_tuple_shape

def test_tpu():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='feynman-tpu', 
        # zone='us-central1-a',
        # project='elegant-matrix-272417'
    )
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    print(strategy)

def main():
    all_graphs = test_graph()
    inputs = [all_graphs[0][0]]
    in_graphs = utils_tf.data_dicts_to_graphs_tuple(inputs)
    in_graphs = utils_tf.set_zero_global_features(in_graphs, 1)
    dtype_shape_from_graphs_tuple(in_graphs)

    # dataset = tf.distribute.Strategy.experimental_make_numpy_dataset(all_graphs)
    # dist_dataset = tf.distribute.Strategy.experimental_distribute_dataset(dataset)
    # print(dist_dataset)

if __name__ == "__main__":
    # test_gen()
    # test_gen_dict()
    # main()
    test_tpu()