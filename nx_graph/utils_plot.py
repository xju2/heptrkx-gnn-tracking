import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

def plot_networkx(G, ax=None, only_true=False):
    """G is networkx graph,
    node feature: {'pos': [r, phi, z]}
    edge feature: {"solution": []}
    """
    if ax is None:
        fig, ax = plt.subplots()

    n_edges = len(G.edges())
    edge_colors = [0.]*n_edges
    true_edges = []
    for iedge,edge in enumerate(G.edges(data=True)):
        if int(edge[2]['solution'][0]) == 1:
            edge_colors[iedge] = 'r'
            true_edges.append((edge[0], edge[1]))
        else:
            edge_colors[iedge] = 'grey' 

    Gp = nx.edge_subgraph(G, true_edges) if only_true else G
    edge_colors = ['r']*len(true_edges) if only_true else edge_colors 

    pos = {}
    for inode, node in enumerate(Gp.nodes()):
        r, phi, z = Gp.node[node]['pos']
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pos[node] = np.array([x, y])

    nx.draw(Gp, pos, node_color='#A0CBE2', edge_color=edge_colors,
       width=0.5, with_labels=False, node_size=1, ax=ax)


def plot_hits(hits, numb=5, fig=None):
    """
    hits is a Dataframe that combines the info from [hits, truth, particles]
    from nx_graph.utils_data import merge_truth_info_to_hits 
    """
    if fig is None:
        fig = plt.figure(figsize=(15, 12))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='polar')

    axs = [ax1, ax2, ax3, ax4]
    particle_ids = np.unique(hits[hits['particle_id']!=0]['particle_id'])
    pID_name = 'particle_id'
    for i in range(numb):
        p = particle_ids[i]
        data = hits[hits[pID_name] == p][['r', 'eta', 'phi', 'z', 'absZ']].sort_values(by=['absZ']).values

        ax1.plot(data[:,3], data[:,0], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax1.scatter(data[:,3], data[:,0], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

        ax2.plot(data[:,3], data[:,1], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax2.scatter(data[:,3], data[:,1], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

        ax3.plot(data[:,3], data[:,2], '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax3.scatter(data[:,3], data[:,2], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)


        ax4.plot(data[:,2], np.abs(data[:,3]), '-', alpha=0.5, lw=4, label='{}'.format(p))
        ax4.scatter(data[:,2], np.abs(data[:,3]), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)

    fontsize=16
    minor_size=14
    y_labels = ['r [mm]', "$\eta$", '$\phi$']
    y_lims = [(0, 1100), (-5, 5), (-np.pi, np.pi)]
    for i in range(3):
        axs[i].set_xlabel('Z [mm]', fontsize=fontsize)
        axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
        axs[i].tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        axs[i].set_xlim(-3200, 3200)
        axs[i].set_ylim(*y_lims[i])

    ax4.grid(True)
    ax4.set_ylim(0, 3200)
    fig.tight_layout()

    return fig, axs

