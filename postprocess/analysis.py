import pandas as pd
import numpy as np

from nx_graph import utils_data

from trackml.score import score_event


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


def summary_on_prediction(G, truth, prediction, do_detail=False):
    """
    truth: DataFrame, contains only good tracks,
    prediction: DataFrame, ['hit_id', 'track_id']
    """
    aa = truth.merge(prediction, on='hit_id', how='inner').sort_values('particle_id')
    ss = aa.groupby('particle_id')[['track_id']].apply(lambda x: len(np.unique(x))) == 1
    wrong_particles = ss[ss == False].index
    n_total_predictions = len(np.unique(prediction['track_id']))
    correct_particles = ss[ss].index
    n_correct = len(correct_particles)
    n_wrong = len(wrong_particles)
    if not do_detail:
        return {
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "correct_pids": correct_particles,
            "wrong_pids": wrong_particles,
            'total_predictions': n_total_predictions
        }

    # are wrong particles due to missing edges (i.e they are isolated)
    connected_pids = []
    isolated_pids = []
    broken_pids = []
    for pp in wrong_particles:
        jj = aa[aa['particle_id'] == pp]

        is_isolated = True
        is_broken = False
        for hit_id in jj['hit_id']:
            node = find_hit_id(G, hit_id)
            if node < 0:
                continue
            nbrss = list(G.neighbors(node))
            if nbrss is None or len(nbrss) < 1:
                is_broken = True
            else:
                is_isolated = False

        if is_isolated:
            isolated_pids.append(pp)

        if is_broken:
            broken_pids.append(pp)
        else:
            connected_pids.append(pp)

    return {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "isolated_pids": isolated_pids,
        "broken_pids": broken_pids,
        "connected_pids": connected_pids,
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
