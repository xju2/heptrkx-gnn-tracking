#!/usr/bin/env python3
import yaml
import os


import torch
import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from datasets.graph import load_graph
from models import get_model

## the glue method
from postprocess import pathfinder
from postprocess import glue

def get_output_dir(config):
    return os.path.expandvars(config['experiment']['output_dir'])

def get_input_dir(config):
    return os.path.expandvars(config['data']['input_dir'])

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)

def load_summaries(config):
    summary_file = os.path.join(get_output_dir(config), 'summaries.npz')
    return np.load(summary_file)

def load_model(config, reload_epoch):
    print('loading model')
    model_config = config['model']
    model_type = model_config.pop('model_type')
    model_config.pop('optimizer', None)
    model_config.pop('learning_rate', None)
    model_config.pop('loss_func', None)
    model = get_model(name=model_type, **model_config)

    # Reload specified model checkpoint
    output_dir = get_output_dir(config)
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model

class TrackingScore(object):
    def __init__(self, config_data_file,
                 config_train_file,
                 method,
                 n_events=1,
                 reload_epoch=18,
                 weight_cut=0.5,
                ):
        config = load_config(config_data_file)
        self.hitgraph_dir = config['output_dir']
        self.trackdata_dir = config['input_dir']
        selection = config['selection']
        self.n_sections = selection['n_phi_sections'] * selection['n_eta_sections']

        # load model from the trained epoch
        self.config_train_file = config_train_file
        self.n_events = n_events
        self.reload_epoch = reload_epoch
        self.model = None
        self.update_model = False
        self.weight_cutoff = weight_cut
        self.method = method

    def print_info(self):
        out =  "# of events: {}\n".format(self.n_events)
        out =  "method:      {}\n".format(self.method)
        out += "weight cut:  {:.2f}\n".format(self.weight_cutoff)
        print(out)

    def set_train_config(self, config_file):
        self.config_train_file = config_file

    def set_n_events(self, nevents):
        self.n_events = nevents

    def load_epoch(self, reload_epoch):
        self.reload_epoch = reload_epoch

    def get_score(self):
        self.print_info()
        if self.update_model or self.model is None:
            self.model = load_model(load_config(self.config_train_file),
                                    reload_epoch=self.reload_epoch).eval()
        all_scores= np.array([
            self.get_score_of_one_event(ievt)
            for ievt in range(self.n_events)
        ])
        return np.mean(all_scores[:, 0]), np.mean(all_scores[:, 1])


    def get_score_of_one_event(self, ievt):
        event_str = "event00000{0}".format(1000+ievt)

        # get score of the track candidates
        # load the event info from tracking ml
        event_input_name = os.path.join(self.trackdata_dir, event_str)
        hits, cells, particles, truth = load_event(event_input_name)

        all_tracks = []
        for i in range(self.n_sections):
            all_tracks += self.get_tracks_of_one_sector(event_str, i, hits, truth)


        # this part takes most of time
        # need improvement
        df_sub = hits[['hit_id']]
        total_tracks = len(all_tracks)
        print("total tracks: ", total_tracks)

        results = []
        for itrk, track in enumerate(all_tracks):
            results += [(x, itrk) for x in track]

        new_df = pd.DataFrame(results, columns=['hit_id', 'track_id'])
        df_sub = df_sub.merge(new_df, on='hit_id', how='outer').fillna(total_tracks+1)
        matched = truth.merge(new_df, on='hit_id', how='inner')
        tot_truth_weight = np.sum(matched['weight'])
        ## remove the hits that belong to the same particle 
        # but of that the total number is less than 50% of the hits of the particle
        particle_ids = np.unique(matched['particle_id'])
        for p_id in particle_ids:
            pID_match = matched[matched['particle_id'] == p_id]
            if pID_match.shape[0] < truth[truth['particle_id'] == p_id].shape[0]*0.5:
                tot_truth_weight -= np.sum(pID_match['weight'])

        return [score_event(truth, df_sub), tot_truth_weight]


    def get_tracks_of_one_sector(self, event_str, iSec, hits, truth):
        file_name = os.path.join(self.hitgraph_dir,
                              event_str+"_g00{0}.npz".format(iSec))
        id_name = file_name.replace('.npz', '_ID.npz')

        G = load_graph(file_name)
        with np.load(id_name) as f:
            hit_ids = f['ID']

        n_hits = G.X.shape[0]
        batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
        with torch.no_grad():
            weights = self.model(batch_input).flatten().numpy()

        return postprocess.glue.get_tracks(G, weights, hit_ids, hits, truth) \
                if self.method=='glue' else \
                postprocess.pathfinder.get_tracks(G, weights, hit_ids, self.weight_cutoff)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Get a score from postprocess')
    add_arg = parser.add_argument
    add_arg('data_config',  nargs='?', default='configs/xy_pre_small.yaml')
    add_arg('train_config', nargs='?', default='configs/hello_graph.yaml')
    add_arg('--nEvents', default=1, type=int, help='number of events for scoring')
    add_arg('--nEpoch',  default=18, type=int, help='reload model from an epoch')
    add_arg('-m', '--method', default='glue', help='postprocess method, [glue, pathfinder]')
    add_arg('--weightCut', default=0.5, type=float, help='weight cut for pathfinder')

    args = parser.parse_args()

    tracking_score = TrackingScore(
        args.data_config,
        args.train_config,
        method=args.method,
        n_events=args.nEvents,
        reload_epoch=args.nEpoch,
        weight_cut=args.weight_cut
    )

    scores = tracking_score.get_score()
    print("score of gnn:   {:.4f}".format(scores[0]))
    print("score of truth: {:.4f}".format(scores[1]))
    print("ratio:          {:.4f}".format(scores[0]/scores[1]))
