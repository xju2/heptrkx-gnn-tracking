#!/usr/bin/env python
"""
Use Denial's triplet graph as an input
"""
import pandas as pd
import numpy as np

from heptrkx.postprocess import wrangler

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Return track candidates from a triplet graph')
    add_arg = parser.add_argument
    add_arg('triplet_file', help='a triplet file contains node and edge info of the triplet graph')
    add_arg('doublet_idx_file', help='the hits that forms the doublet')
    args = parser.parse_args()

    df_triplet = pd.read_csv(args.triplet_file, header=None, names=['evtid', 'd1', 'd2', 'score'])
    edge_feature = 'solution'

    #df.score.plot(kind='hist', logy=True, bins=50)

    nodes = np.unique(np.concatenate((df_triplet.d1.values, df_triplet.d2.values)))
    edges = [(e1, e2, {edge_feature: [v]}) for e1, e2, v in zip(
        df_triplet.d1.values, df_triplet.d2.values, df_triplet.score.values
        )]
    G3 = nx.Graph() # triplet graph
    G3.add_nodes_from(nodes)
    G3.add_edges_from(edges)

    tracks = wrangler.get_tracks(G, th=0.99, th_re=0.999, feature_name=edge_feature, with_fit=False)
    
    results = []
    for itrk, track in enumerate(nx_graphs):
        results += [(track.node[x], itrk) for x in track.nodes()]

    df_tracks = pd.DataFrame(results, columns=['node_id', 'track_id'])
    print(df_tracks.head())