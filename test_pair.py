#!/usr/bin/env python

from nx_graph.utils_test import create_trained_model
from nx_graph.utils_test import plot_metrics

if __name__ == "__main__":
    config_file = 'configs/nxgraph_test_pairs.yaml'
    input_ckpt = 'trained_results/nxgraph_pairs_4evts/bak'
    model = create_trained_model(config_file, input_ckpt)
    print("HERE")

    iteration = 15998
    n_graphs = 200 ## batch-size = 2
    odd, tdd = model(iteration, n_graphs)
    plot_metrics(odd, tdd, odd_th=0.5)
