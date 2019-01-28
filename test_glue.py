#!/usr/bin/env python3
from glue import  create_glue
from glue import  n_det_layers

import numpy as np
import pandas as pd
import torch
import time

if __name__ == "__main__":
    from datasets.graph import load_graph
    file_name = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_001/event000001000_g000.npz'
    id_name   = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_002/event000001000_g000_ID.npz'

    G = load_graph(file_name)
    with np.load(id_name) as f:
        hit_ids = f['ID']

    from score import load_model
    from score import load_config
    model_config_file = 'configs/hello_graph.yaml'

    model = load_model(load_config(model_config_file), reload_epoch=18).eval()
    batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
    with torch.no_grad():
        weights = model(batch_input).flatten().numpy()

    print("precision: ", np.sum(weights*G.y)/G.y.nonzero()[0].shape[0])


    event_input_name = '/global/cscratch1/sd/xju/heptrkx/trackml_inputs/train_all/event000001000'
    from trackml.dataset import load_event
    hits, cells, particles, truth = load_event(event_input_name)

    glue_func = create_glue(G, weights, hit_ids, hits, truth)
    start_time = time.time()
    outer_layer = n_det_layers - 1
    inner_layer = outer_layer - 1
    layer_pairs, precision = glue_func(outer_layer, inner_layer)
    end_time = time.time()
    print("{} to {} takes: {:.1f} ms with precision {:.4f}".format(
        outer_layer, inner_layer, (end_time - start_time)*1000, precision)
    )
