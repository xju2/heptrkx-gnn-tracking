from postprocess.evaluate_tf import create_evaluator

import numpy as np

from graph_nets import utils_np
from nx_graph import prepare, utils_test


config_file = 'configs/nxgraph_test_pairs.yaml'
input_ckpt = 'trained_results/nxgraph_pairs_004'

model = create_evaluator(config_file, 99987, input_ckpt)

is_digraph = True
is_bidirection = False
file_name = '/global/cscratch1/sd/xju/heptrkx/data/graphs_from_pairs_phi4_eta6/event000001099_g000000000_INPUT.npz'
input_graphs = [prepare.load_data_dicts(file_name)]
target_graphs = [prepare.load_data_dicts(file_name.replace("INPUT", "TARGET"))]

graphs = model(utils_np.data_dicts_to_graphs_tuple(input_graphs),
               utils_np.data_dicts_to_graphs_tuple(target_graphs),
               use_digraph=is_digraph, bidirection=is_bidirection
              )

G = graphs[0]
weight = [G.edges[edge]['predict'][0] for edge in G.edges()]
truth  = [G.edges[edge]['solution'][0] for edge in G.edges()]

weights = np.array(weight)
truths = np.array(truth)
utils_test.plot_metrics(weights, truths, odd_th=0.1)
