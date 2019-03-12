
from .utils_fit import poly_fit_phi
from .utils_fit import pairwise

import networkx as nx
import numpy as np

from functools import partial

def get_nbr_weights(G, pp, used_hits=None, feature_name='solution', th=0.1):
    nbrs = list(set(nx.neighbors(G, pp)).difference(set(used_hits)))
    if len(nbrs) < 1:
        return None,None

    weights = [G.edges[(pp, i)][feature_name][0] for i in nbrs]
    if max(weights) < th:
        return None,None

    sort_idx = list(reversed(np.argsort(weights)))
    nbrss = [nbrs[x] for x in sort_idx]
    wss = [weights[x] for x in sort_idx]

    return nbrss, wss


def find_next_hits(G, pp, used_hits, th=0.1, th_re=0.8, feature_name='solution'):
    """G is the graph, path is previous hits."""

    nbrs = list(set(nx.neighbors(G, pp)).difference(set(used_hits)))
    if len(nbrs) < 1:
        return None

    weights = [G.edges[(pp, i)][feature_name][0] for i in nbrs]
    if max(weights) < th:
        return None

    sorted_idx = list(reversed(np.argsort(weights)))
    next_hits = [nbrs[sorted_idx[0]]]
    for ii in range(1, len(sorted_idx)):
        idx = sorted_idx[ii]
        w = weights[idx]
        if w > th_re:
            next_hits.append(nbrs[idx])
        else:
            break

    return next_hits


def build_roads(G, ss, next_hit_fn, used_hits):
    """
    next_hit_fn: a function return next hits, could be find_next_hits
    """
    # get started
    next_hits = next_hit_fn(G, ss, used_hits)
    if next_hits is None:
        return [(ss,None)]
    path = []
    for hit in next_hits:
        path.append((ss, hit))

    while True:
        new_path = []
        is_all_none = True
        for pp in path:
            if pp[-1] is not None:
                is_all_none = False
                break
        if is_all_none:
            break

        for pp in path:
            start = pp[-1]
            if start is None:
                new_path.append(pp)
                continue

            used_hits_cc = np.unique(used_hits + list(pp))
            next_hits = next_hit_fn(G, pp[-1], used_hits_cc)
            if next_hits is None:
                new_path.append(pp + (None,))
            else:
                for hit in next_hits:
                    new_path.append(pp + (hit,))

        path = new_path
    return path


def fit_road(G, road):
    """use a linear function to fit phi as a function of z."""
    road_chi2 = []
    for path in road:
        z   = np.array([G.node[i]['pos'][2] for i in path[:-1]])
        phi = np.array([G.node[i]['pos'][1] for i in path[:-1]])
        if len(z) > 1:
            _, _, diff = poly_fit_phi(z, phi)
            road_chi2.append(np.sum(diff)/len(z))
        else:
            road_chi2.append(1)

#         print(chi2)
    return road_chi2



def chose_a_road(road, diff):
    res = road[0]
    # only if another road has small difference in phi-fit
    # and longer than the first one, it is used.
    for i in range(1, len(road)):
        if diff[i] < diff[0] and len(road[i]) > len(res):
            res = road[i]

    return res



def get_tracks(G, th=0.1, th_re=0.8, feature_name='solution', with_fit=True):
    used_nodes = []
    sub_graphs = []
    next_hit_fn = partial(find_next_hits, th=th, th_re=th_re, feature_name=feature_name)
    for node in G.nodes():
        if node in used_nodes:
            continue
        road = build_roads(G, node, next_hit_fn, used_nodes)
        diff = fit_road(G, road) if with_fit else [0.]*len(road)
        a_road = chose_a_road(road, diff)

        if len(a_road) < 3:
            used_nodes.append(node)
            continue

        a_track = list(pairwise(a_road[:-1]))
        sub = nx.edge_subgraph(G, a_track)
        sub_graphs.append(sub)
        used_nodes += list(sub.nodes())

    return sub_graphs
