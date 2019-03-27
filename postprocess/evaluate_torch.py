"""Take a model configation and evaluate hitsgraph"""

import torch
from models import get_model
from datasets.graph import load_graph
from nx_graph.utils_data import hitsgraph_to_networkx
from nx_graph.utils_data import correct_networkx

import re
import glob
import yaml
import os

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)


def load_model(config, reload_epoch):
    model_config = config['model']
    model_type = model_config.pop('model_type')
    model_config.pop('optimizer', None)
    model_config.pop('learning_rate', None)
    model_config.pop('loss_func', None)
    model = get_model(name=model_type, **model_config)

    # Reload specified model checkpoint
    output_dir = os.path.expandvars(config['experiment']['output_dir'])
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model


def create_evaluator(config_file, reload_epoch):
    """use training configrations to initialize models,
    return a function that could evaluate any event, or event section
    """
    config = load_config(config_file)
    model = load_model(config, reload_epoch).eval()
    hitsgraph_dir = config['data']['input_dir']
    base_dir = os.path.join(hitsgraph_dir, 'event{:09d}_g{:03d}.npz')

    def evaluate(evtid, isec=-1, use_digraph=False, bidirection=False):
        """for a given event ID, return a list of graphs whose edges have a feature of __predict__
        and a feature of __solution__
        """
        if isec < 0:
            section_patten = base_dir.format(evtid, 0).replace('_g000', '*')
            n_sections = int(len(glob.glob(section_patten)))
            file_names = [base_dir.format(evtid, ii) for ii in range(n_sections)]
        else:
            file_names = [base_dir.format(evtid, isec)]

        graphs = []
        for file_name in file_names:
            G = load_graph(file_name)
            batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
            with torch.no_grad():
                weights = model(batch_input).flatten().numpy()

            nx_G = hitsgraph_to_networkx(G, use_digraph=use_digraph,
                                               bidirection=bidirection)
            ## update edge features with the new weights
            n_nodes, n_edges = G.Ri.shape
            for iedge in range(n_edges):
                in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
                out_node_id = G.Ro[:, iedge].nonzero()[0][0]
                try:
                    nx_G.edges[(out_node_id, in_node_id)]['predict'] = [weights[iedge]]
                    if use_digraph and bidirection:
                        nx_G.edges[(in_node_id, out_node_id)]['predict'] = [weights[iedge]]
                except KeyError:
                    pass

            graphs.append(nx_G)
        return graphs

    return evaluate
