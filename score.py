#!/usr/bin/env python
import yaml
import os


import torch
import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from datasets.graph import load_graph
from models import get_model

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
                 n_events=1,
                 reload_epoch=18
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


    def set_train_config(self, config_file):
        self.config_train_file = config_file


    def set_n_events(self, nevents):
        self.n_events = nevents

    def load_epoch(self, reload_epoch):
        self.reload_epoch = reload_epoch

    def get_score(self):
        if self.update_model or self.model is None:
            self.model = load_model(load_config(self.config_train_file),
                                    reload_epoch=self.reload_epoch).eval()
        all_scores= [
            self.get_score_of_one_event(ievt)
            for ievt in range(self.n_events)
        ]
        return np.mean(all_scores)


    def get_score_of_one_event(self, ievt):
        event_str = "event00000{0}".format(1000+ievt)

        all_tracks = []
        for i in range(self.n_sections):
            all_tracks += self.get_tracks_of_one_sector(event_str, i)

        # get score of the track candidates
        # load the event info from tracking ml
        event_input_name = os.path.join(self.trackdata_dir, event_str)
        hits, cells, particles, truth = load_event(event_input_name)

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

        return score_event(truth, df_sub)


    def get_tracks_of_one_sector(self, event_str, iSec):
        file_name = os.path.join(self.hitgraph_dir,
                              event_str+"_g00{0}.npz".format(iSec))
        id_name = file_name.replace('.npz', '_ID.npz')

        G = load_graph(file_name)
        with np.load(id_name) as f:
            hit_ids = f['ID']

        n_hits = G.X.shape[0]
        batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
        with torch.no_grad():
            test_outputs = self.model(batch_input).flatten()

        hits_in_tracks = []
        hits_idx_in_tracks = []
        all_tracks = []
        weights = test_outputs.numpy()
        for idx in range(n_hits):
            # Loop over all hits 
            # and save hits that are used in a track
            hit_id = hit_ids[idx]
            if hit_id not in hits_in_tracks:
                hits_in_tracks.append(hit_id)
                hits_idx_in_tracks.append(idx)
            else:
                continue

            a_track = [hit_id]
            while(True):
                # for this hit index (idx),
                # find its outgoing hits that could form a track
                hit_out = G.Ro[idx]
                if hit_out.nonzero()[0].shape[0] < 1:
                    break
                weighted_outgoing = np.argsort((hit_out * weights))
                ii = -1
                has_next_hit = False
                while abs(ii) < 15:
                    weight_idx = weighted_outgoing[ii]
                    next_hit = G.Ri[:, weight_idx].nonzero()
                    if next_hit[0].shape[0] > 0:
                        next_hit_id = next_hit[0][0]
                        if next_hit_id != idx and next_hit_id not in hits_idx_in_tracks:
                            hits_in_tracks.append(hit_ids[next_hit_id])
                            hits_idx_in_tracks.append(next_hit_id)
                            a_track       .append(hit_ids[next_hit_id])
                            idx = next_hit_id
                            has_next_hit = True
                            break
                    ii -= 1

                if not has_next_hit:
                    # no more out-going tracks
                    break
            all_tracks    .append(a_track)
        return all_tracks




if __name__ == "__main__":
    tracking_score = TrackingScore(
        'configs/xy_pre_small.yaml',
        'configs/hello_graph.yaml',
        n_events=1,
        reload_epoch=18)
    print("mean score:", tracking_score.get_score())
