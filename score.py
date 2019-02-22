#!/usr/bin/env python3
import yaml
import os
import logging


import torch
import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from datasets.graph import load_graph
from models import get_model

## the tracking methods
from postprocess import pathfinder
from postprocess import glue

def get_output_dir(config):
    return os.path.expandvars(config['experiment']['output_dir'])

def get_input_dir(config):
    return os.path.expandvars(config['data']['input_dir'])

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)

def print_config(config_file):
    out = "------{}------\n".format(config_file)
    with open(config_file) as f:
        for line in f:
            out += line
    out += "--------------\n"
    return out


def load_summaries(config):
    summary_file = os.path.join(get_output_dir(config), 'summaries.npz')
    return np.load(summary_file)

def load_model(config, reload_epoch):
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

def score_tracks(all_tracks, hits, truth):
    # this part takes most of time
    # need improvement
    total_tracks = len(all_tracks)
    # logging.info("total tracks: {}".format(total_tracks))

    results = []
    for itrk, track in enumerate(all_tracks):
        results += [(x, itrk) for x in track]

    new_df = pd.DataFrame(results, columns=['hit_id', 'track_id'])
    new_df = new_df.drop_duplicates(subset='hit_id')

    df_sub = hits[['hit_id']]
    df_sub = df_sub.merge(new_df, on='hit_id', how='outer').fillna(total_tracks+1)
    matched = truth.merge(new_df, on='hit_id', how='inner')
    tot_truth_weight = np.sum(matched['weight'])

    ## remove the hits that belong to the same particle
    # but of that the total number is less than 50% of the hits of the particle
    particle_ids = np.unique(matched['particle_id'])
    for p_id in particle_ids:
        pID_match = matched[matched['particle_id'] == p_id]
        if pID_match.shape[0] <= truth[truth['particle_id'] == p_id].shape[0]*0.5:
            tot_truth_weight -= np.sum(pID_match['weight'])

    return [score_event(truth, df_sub), tot_truth_weight]


class TrackingScore(object):
    def __init__(self, config_data_file,
                 config_train_file,
                 method,
                 reload_epoch=18,
                 weight_cut=0.5,
                ):
        #logging.debug(print_config(config_data_file ))
        #logging.debug(print_config(config_train_file))

        config = load_config(config_data_file)
        self.hitgraph_dir = config['output_dir']
        self.trackdata_dir = config['input_dir']
        selection = config['selection']
        self.n_sections = selection['n_phi_sections'] * selection['n_eta_sections']

        # load model from the trained epoch
        self.config_train_file = config_train_file
        self.reload_epoch = reload_epoch
        self.model = None
        self.update_model = False
        self.weight_cutoff = weight_cut
        self.method = method

    def print_info(self):
        out =  "method:      {}\n".format(self.method)
        out += "weight cut:  {:.2f}\n".format(self.weight_cutoff)
        logging.info(out)

    def set_train_config(self, config_file):
        self.config_train_file = config_file

    def load_epoch(self, reload_epoch):
        self.reload_epoch = reload_epoch

    def get_score(self, n_events):
        self.print_info()
        if self.update_model or self.model is None:
            self.model = load_model(load_config(self.config_train_file),
                                    reload_epoch=self.reload_epoch).eval()
        all_scores= np.array([
            self.get_score_of_one_event(ievt)
            for ievt in range(n_events)
        ])
        return np.mean(all_scores[:, 0]), np.mean(all_scores[:, 1])


    def get_score_of_one_event(self, ievt):
        event_str = "event00000{0}".format(1000+ievt)
        logging.debug("processing {} event:".format(event_str))

        # get score of the track candidates
        # load the event info from tracking ml
        event_input_name = os.path.join(self.trackdata_dir, event_str)
        hits, truth = load_event(event_input_name, parts=['hits', 'truth'])

        all_gnn_tracks = []
        all_true_tracks = []
        for i in range(self.n_sections):
            res_tracks = self.get_tracks_of_one_sector(event_str, i, hits, truth)
            all_gnn_tracks += res_tracks[0]
            all_true_tracks += res_tracks[1]

        event_gnn   = score_tracks(all_gnn_tracks,  hits, truth)
        event_truth = score_tracks(all_true_tracks, hits, truth)
        logging.debug("SCORE of {} event: {:.4f} {:.4f} {:.4f}, {:.4f}\n".format(
                      ievt, event_gnn[0], event_truth[0], event_gnn[0]/event_truth[0], event_truth[1])
        )
        return [event_gnn[0], event_truth[0]]


    def get_tracks_of_one_sector(self, event_str, iSec, hits, truth):
        logging.debug("processing {} section".format(iSec))
        file_name = os.path.join(self.hitgraph_dir,
                                 event_str+"_g{:03d}.npz".format(iSec))
        id_name = file_name.replace('.npz', '_ID.npz')

        G = load_graph(file_name)
        with np.load(id_name) as f:
            hit_ids = f['ID']

        n_hits = G.X.shape[0]
        batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
        with torch.no_grad():
            weights = self.model(batch_input).flatten().numpy()

        gnn_tracks = glue.get_tracks(G, weights, hit_ids, hits, truth) \
                if self.method=='glue' else \
                pathfinder.get_tracks(G, weights, hit_ids, self.weight_cutoff)

        true_tracks = glue.get_tracks(G, G.y, hit_ids, hits, truth) \
                if self.method=='glue' else \
                pathfinder.get_tracks(G, G.y, hit_ids, self.weight_cutoff)

        return gnn_tracks, true_tracks


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Get a score from postprocess')
    add_arg = parser.add_argument
    add_arg('data_config',  nargs='?', default='configs/xy_pre_small.yaml')
    add_arg('train_config', nargs='?', default='configs/hello_graph.yaml')
    add_arg('--nEvents',    default=1, type=int, help='number of events for scoring')
    add_arg('--nEpoch',     default=18, type=int, help='reload model from an epoch')
    add_arg('-m', '--method', default='glue', help='postprocess method, [glue, pathfinder]')
    add_arg('--weightCut', default=0.5, type=float, help='weight cut for pathfinder')
    add_arg('--log', default='info', help='logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(filename='score.log',
                        level=numeric_level,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    tracking_score = TrackingScore(
        args.data_config,
        args.train_config,
        method=args.method,
        reload_epoch=args.nEpoch,
        weight_cut=args.weightCut
    )

    scores = tracking_score.get_score(args.nEvents)
    logging.info("score of gnn:   {:.4f}".format(scores[0]))
    logging.info("score of truth: {:.4f}".format(scores[1]))
    logging.info("ratio:          {:.4f}".format(scores[0]/scores[1]))
