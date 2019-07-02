import pandas as pd
import numpy as np

from heptrkx.nx_graph import utils_data

from trackml.score import score_event
from trackml.score import _analyze_tracks as analyze_tracks


def find_hit_id(G, idx):
    res = -1
    for node in G.nodes():
        if G.node[node]['hit_id'] == idx:
            res = node
            break
    return res


def get_nbr_weights(G, pp, weight_name='solution'):
    nbrs = list(nx.neighbors(G, pp))
    if nbrs is None or len(nbrs) < 1:
        return None,None

    weights = [G.edges[(pp, i)][weight_name][0] for i in nbrs]
    sort_idx = list(reversed(np.argsort(weights)))
    nbrss = [nbrs[x] for x in sort_idx]
    wss = [weights[x] for x in sort_idx]
    return nbrss, wss


def incoming_nodes_hitsgraph(G, node_id):
    # current node is considered as outgoing in an edge, find incoming nodes
    return [np.nonzero(G.Ri[:, ii])[0][0] for ii in np.nonzero(G.Ro[node_id, :])[0]]


def outgoing_nodes_hitsgraph(G, node_id):
    # current node as incoming in an edge, find outgoing nodes
    outgoing_nodes = [np.nonzero(G.Ro[:, ii])[0][0] for ii in np.nonzero(G.Ri[node_id, :])[0]]


def graphs_to_df(nx_graphs):
    results = []
    for itrk, track in enumerate(nx_graphs):
        results += [(track.node[x]['hit_id'], itrk) for x in track.nodes()]

    df = pd.DataFrame(results, columns=['hit_id', 'track_id'])
    # new_df = new_df.drop_duplicates(subset='hit_id')
    return df


def summary_on_prediction(G, truth, prediction, matching_cut=0.0):
    """Find number of track candidates that can match to a true track.
    The matching requires the track candidate contains at least *mathching_cut*
    percentange of hits from the true track.

    G -- graph, that contains all hit and edge info,
    truth -- DataFrame, contains true tracks,
    prediction -- DataFrame, ['hit_id', 'track_id'].
    matching_cut -- percentage of hits from the true track
                    that are contained in the track candidate
    """
    truth_hit_id = truth[truth.hit_id.isin(prediction.hit_id)]
    tracks = analyze_tracks(truth_hit_id, prediction)
    print("Track ML score: ", score_event(truth_hit_id, prediction))
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_partcile_nhits'])
    matched_tracks = tracks[purity_maj > matching_cut]

    n_total_predictions = tracks.shape[0]
    correct_particles = matched_tracks.major_partcile_id.to_numpy()
    n_correct = matched_tracks.shape[0]
    n_wrong = n_total_predictions - n_correct
    return {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "correct_pids": correct_particles,
        "wrong_pids": wrong_particles,
        'total_predictions': n_total_predictions
    }


def label_particles(hits, truth, threshold=1.0, ignore_noise=True):
    """
    hits only include a subset of hits in truth.
    Threshold defines the threshold value on the fraction of hits in used
    that are associated with a particle over all hits associated with the particle
    1 requires all hits in a track are in hits
    0.5 requires a half of track are in hits
    """
    ## particles in question
    piq= hits.merge(truth, on='hit_id', how='left')

    pids = np.unique(piq['particle_id'])
    bad_pids = []
    good_pids = []
    for p_id in pids:
        hits_from_reco = piq[piq['particle_id'] == p_id]
        hits_from_true = truth[truth['particle_id'] == p_id]
        w_used_hits = np.sum(hits_from_reco['weight'])
        if ignore_noise and w_used_hits < 1e-7:
            continue

        if hits_from_reco.shape[0] < hits_from_true.shape[0]*threshold:
            bad_pids.append(p_id)
        else:
            good_pids.append(p_id)
    return good_pids, bad_pids


def score_nxgraphs(nx_graphs, truth):
    """nx_graphs: a list of networkx graphs, with node feature hit_id
    each graph is a track"""
    total_tracks = len(nx_graphs)

    new_df = graphs_to_df(nx_graphs)

    results = []
    for itrk, track in enumerate(nx_graphs):
        results += [(track.node[x]['hit_id'], itrk) for x in track.nodes()]

    new_df = pd.DataFrame(results, columns=['hit_id', 'track_id'])

    #df_sub = hits[['hit_id']]
    #df_sub = df_sub.merge(new_df, on='hit_id', how='outer').fillna(total_tracks+1)
    matched = truth.merge(new_df, on='hit_id', how='inner')
    tot_truth_weight = np.sum(matched['weight'])

    ## remove the hits if their total is less than 50% of the hits
    # that belong to the same particle.
    reduced_weights = 0
    w_broken_trk = 0
    particle_ids = np.unique(matched['particle_id'])
    for p_id in particle_ids:
        hits_from_reco = matched[matched['particle_id'] == p_id]
        hits_from_true = truth[truth['particle_id'] == p_id]
        w_used_hits = np.sum(hits_from_reco['weight'])
        if hits_from_reco.shape[0] <= hits_from_true.shape[0]*0.5:
            reduced_weights += w_used_hits
        if hits_from_reco.shape[0] != hits_from_true.shape[0]:
            w_broken_trk += w_used_hits

    return [score_event(truth, new_df), tot_truth_weight, reduced_weights, w_broken_trk]


def trk_eff_purity(true_tracks, predict_tracks):
    n_same = 0
    true_ones = []
    for true_track in true_tracks:
        is_same = False
        for tt in predict_tracks:
            if not utils_data.is_diff_networkx(true_track, tt):
                is_same = True
                break
        if is_same :
            true_ones.append(true_track)
            n_same += 1
    eff = n_same * 1./len(true_tracks)

    n_true = 0
    fake_ones = []
    for track in predict_tracks:
        is_true = False
        for tt in true_tracks:
            if not utils_data.is_diff_networkx(track, tt):
                is_true = True
                break
        if is_true:
            n_true += 1
        else:
            fake_ones.append(track)
    purity = n_true*1./len(predict_tracks)
    return eff, purity, true_ones, fake_ones


def eff_vs_pt(true_predicts, true_tracks):
    sel_pt  = [track.node[list(track.nodes())[0]]['info'][0,0] for track in true_predicts]
    true_pt = [track.node[list(track.nodes())[0]]['info'][0,0] for track in true_tracks]
    return sel_pt, true_pt


def mistagged_edges(graph, threshold=0.1):
    res_edges = [edge for edge in graph.edges() if graph.edges[edge]['solution']==1 and graph.edges[edge]['predict'] < threshold]
    subgraph = graph.edge_subgraph(res_edges)
    return subgraph



def fake_edges(graph):
    """graph is fake graph, defined as hits in the track are different from true ones"""
    n_edges = len(graph.edges())
    n_fake_edges = len([edge for edge in graph.edges() if graph.edges[edge]['solution'] == 1])
    return n_fake_edges, n_fake_edges/n_edges, n_edges


def inspect_events(hits, particles, truth, min_hits=3):
    n_hits = hits.shape[0]
    n_p = particles.shape[0]
    hits_truth = hits.merge(truth, on='hit_id', how='left')
    n_noise_hits = hits_truth[hits_truth.particle_id == 0].shape[0]

    # number of detectable particles
    particle_hits = particles.merge(hits_truth, on='particle_id', how='left')
    n_dp = particle_hits[np.isnan(particle_hits.hit_id)].shape[0]

    # number of particles with at least three hits
    dp =  particle_hits[~np.isnan(particle_hits.hit_id)]
    good_particles = dp.groupby('particle_id')['hit_id'].count() > min_hits-1
    n_good_p = np.sum(good_particles)

    def pp():
        print("# of hits: ", n_hits)
        print("# of noise hits: {} ({:.1f}%)".format(n_noise_hits, 100.*n_noise_hits/n_hits))
        print("# of particles: {}".format(n_p))
        print("# of detectable particles: {} ({:.1f}%)".format(n_dp, 100.*n_dp/n_p))
        print("# of good particles: {} ({:.1f}%)".format(n_good_p, 100.*n_good_p/n_p))

    return n_hits, n_noise_hits, n_p, n_dp, n_good_p, good_particles.index.to_numpy(), pp
