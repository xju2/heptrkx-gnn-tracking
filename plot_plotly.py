#!/usr/bin/env python3
"""
Example from: https://plot.ly/python/network-graphs/
"""
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio

import networkx as nx
import numpy as np
import os

from datasets.graph import load_graph
from nx_graph.prepare import hitsgraph_to_networkx_graph

def show_nx_graph(G, filename='test_networkx', is_3d=False, dynamic=False):
    pos = nx.get_node_attributes(G, 'pos')

    dmin = 1
    ncenter = 0
    for n in pos:
        r, phi, z = pos[n]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        d = (x-0.5)**2 + (y-0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d
    p = nx.single_source_shortest_path_length(G, ncenter)

    n_edges = len(G.edges())
    n_nodes = len(G.nodes())
    print("add ", n_edges, "egdes")
    print("add ", n_nodes, "nodes")

    edge_x_fn = [0.]*n_edges
    edge_y_fn = [0.]*n_edges
    edge_z_fn = [0.]*n_edges
    true_x_fn = []
    true_y_fn = []
    true_z_fn = []
    iedge = 0
    for in_node, out_node, features in G.edges(data=True):
        r0, phi0, z0 = G.node[in_node]['pos']
        r1, phi1, z1 = G.node[out_node]['pos']
        x0 = r0*np.cos(phi0)
        y0 = r0*np.sin(phi0)
        x1 = r1*np.cos(phi1)
        y1 = r1*np.sin(phi1)
        edge_x_fn[iedge] = [x0, x1, None]
        edge_y_fn[iedge] = [y0, y1, None]
        edge_z_fn[iedge] = [z0, z1, None]
        iedge += 1
        if features['solution'] == 1:
            true_x_fn.append([x0, x1, None])
            true_y_fn.append([y0, y1, None])
            true_z_fn.append([z0, z1, None])

    # covert to tuple
    def flattern(full_list):
        return [item for sublist in full_list for item in sublist]

    edge_x_fn = tuple(flattern(edge_x_fn))
    edge_y_fn = tuple(flattern(edge_y_fn))
    edge_z_fn = tuple(flattern(edge_z_fn))
    true_x_fn = tuple(flattern(true_x_fn))
    true_y_fn = tuple(flattern(true_y_fn))
    true_z_fn = tuple(flattern(true_z_fn))

    if is_3d:
        edge_trace = go.Scatter3d(
            x=edge_x_fn,
            y=edge_y_fn,
            z=edge_z_fn,
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')
        true_edge_trace = go.Scatter3d(
            x=true_x_fn,
            y=true_y_fn,
            z=true_z_fn,
            line=dict(width=0.5,color='rgb(205, 12, 24)'),
            hoverinfo='none',
            mode='lines')
    else:
        edge_trace = go.Scatter(
            x=edge_x_fn,
            y=edge_y_fn,
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')
        true_edge_trace = go.Scatter(
            x=true_x_fn,
            y=true_y_fn,
            line=dict(width=0.5,color='rgb(205, 12, 24)'),
            hoverinfo='none',
            mode='lines')


    node_x_fn = [0.]*n_nodes
    node_y_fn = [0.]*n_nodes
    node_z_fn = [0.]*n_nodes
    inode = 0
    for node in G.nodes():
        r, phi, z = G.node[node]['pos']
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        node_x_fn[inode] = [x]
        node_y_fn[inode] = [y]
        node_z_fn[inode] = [z]
        inode += 1
    node_x_fn = tuple(flattern(node_x_fn))
    node_y_fn = tuple(flattern(node_y_fn))
    node_z_fn = tuple(flattern(node_z_fn))

    node_trace = go.Scatter3d(
        x=node_x_fn,
        y=node_y_fn,
        z=node_z_fn,
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=2,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    # Color node points by the number of connections
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = '# of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])

    # create network graph
    fig = go.Figure(data=[edge_trace, true_edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Hit connections viewed in x-y plane',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="High density tracking data",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    if dynamic:
        py.iplot(fig, filename=filename)
    else:
        if not os.path.exists('images'):
            os.mkdir('images')
        pio.write_image(fig, os.path.join('images', filename+".png"))





if __name__ == "__main__":
    hits_graph_dir = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_big_000'
    nx_graph_dir = '/global/cscratch1/sd/xju/heptrkx/data/nxgraphs_big_000'
    hits_graph_name = 'event000001001_g015.npz'

    G = hitsgraph_to_networkx_graph(load_graph(
        os.path.join(hits_graph_dir, hits_graph_name)))

    #show_nx_graph(G, filename='test_networkx', is_3d=False)
    show_nx_graph(G, filename='test_networkx_3d', is_3d=True)
