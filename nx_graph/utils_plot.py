import networkx as nx
import numpy as np

def plot_networkx(G, ax=None):
    """G is networkx graph,
    node feature: {'pos': [r, phi, z]}
    edge feature: {"solution": []}
    """
    n_edges = len(G.edges())
    edge_colors = [0.]*n_edges
    for iedge,edge in enumerate(G.edges(data=True)):
        edge_colors[iedge] = 'r' if int(edge[2]['solution'][0]) == 1 else 'grey'

    pos = {}
    for inode, node in enumerate(G.nodes()):
        r, phi, z = G.node[node]['pos']
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pos[inode] = np.array([x, y])

    nx.draw(G, pos, node_color='#A0CBE2', edge_color=edge_colors,
       width=0.5, with_labels=False, node_size=1, ax=ax)
