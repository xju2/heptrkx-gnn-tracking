import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

import sklearn.metrics

from more_itertools import pairwise

fontsize=16
minor_size=14

def get_pos(Gp):
    pos = {}
    for node in Gp.nodes():
        r, phi, z = Gp.node[node]['pos'][:3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pos[node] = np.array([x, y])
    return pos


def plot_networkx(G, ax=None, only_true=False):
    """G is networkx graph,
    node feature: {'pos': [r, phi, z]}
    edge feature: {"solution": []}
    """
    if ax is None:
        _, ax = plt.subplots()

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
    edge_colors = ['k']*len(true_edges) if only_true else edge_colors 

    pos = get_pos(Gp)

    nx.draw(Gp, pos, node_color='#A0CBE2', edge_color=edge_colors,
       width=0.5, with_labels=False, node_size=1, ax=ax, arrows=False)



def plot_nx_with_edge_cmaps(
    G, weight_name='predict', weight_range=(0, 1),
    ax=None, cmaps=plt.get_cmap('Greys'), threshold=0.):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    pos = get_pos(G)
    #edges, weights = zip(*nx.get_edge_attributes(G, weight_name).items())
    #weights = [x[0] for x in weights]
    res = [(edge, G.edges[edge]['predict'][0]) for edge in G.edges() if G.edges[edge]['predict'][0] > threshold]
    edges, weights = zip(*dict(res).items())

    vmin, vmax = weight_range

    nx.draw(G, pos, node_color='#A0CBE2', edge_color=weights, edge_cmap=cmaps,
            edgelist=edges, width=0.5, with_labels=False,
            node_size=1, edge_vmin=vmin, edge_vmax=vmax, ax=ax, arrows=False)


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


def plot_metrics(odd, tdd, odd_th=0.5, tdd_th=0.5, outname='roc_graph_nets.eps', off_interactive=False):
    if off_interactive:
        plt.ioff()

    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    accuracy  = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall    = sklearn.metrics.recall_score(y_true, y_pred)

    print('Accuracy:            %.6f' % accuracy)
    print('Precision (purity):  %.6f' % precision)
    print('Recall (efficiency): %.6f' % recall)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)


    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    ax0, ax1, ax2, ax3 = axs

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning=dict(bins=50, histtype='step', log=True)
    ax0.hist(odd[y_true==False], lw=2, label='fake', **binning)
    ax0.hist(odd[y_true], lw=2, label='true', **binning)
    ax0.set_xlabel('Model output', fontsize=fontsize)
    ax0.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax0.legend(loc=0, fontsize=fontsize)

    # Plot the ROC curve
    auc = sklearn.metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], '--', lw=2)
    ax1.set_xlabel('False positive rate', fontsize=fontsize)
    ax1.set_ylabel('True positive rate', fontsize=fontsize)
    ax1.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)
    ax1.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    p, r, t = sklearn.metrics.precision_recall_curve(y_true, odd)
    ax2.plot(t, p[:-1], label='purity', lw=2)
    ax2.plot(t, r[:-1], label='efficiency', lw=2)
    ax2.set_xlabel('Cut on model score', fontsize=fontsize)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.legend(fontsize=fontsize, loc='upper right')

    ax3.plot(p, r, lw=2)
    ax3.set_xlabel('Purity', fontsize=fontsize)
    ax3.set_ylabel('Efficiency', fontsize=fontsize)
    ax3.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    plt.savefig(outname)
    if off_interactive:
        plt.close(fig)


def pixel_matrix(pixel_cluster, show=False):
    # cluster size
    min0 = min(pixel_cluster['ch0'])
    max0 = max(pixel_cluster['ch0'])
    min1 = min(pixel_cluster['ch1'])
    max1 = max(pixel_cluster['ch1'])
    # the matrix
    matrix = np.zeros(((max1-min1+3),(max0-min0+3)))
    for pixel in pixel_cluster.values :
        i0 = int(pixel[1]-min0+1)
        i1 = int(pixel[2]-min1+1)
        value = pixel[3]
        matrix[i1][i0] = value 
    # return the matris
    if show :
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('YlOrRd'))
        plt.colorbar()
        plt.show()
    return matrix, max0-min0+1, max1-min1+1


def plot_ratio(tot, sel, label_tot, label_sel,
                    xlabel, title, outname, **plot_options):
    plt.clf()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios':[4, 1]})
    fig.subplots_adjust(hspace=0)

    val_tot, bins, _ = ax0.hist(tot, label=label_tot, **plot_options)
    val_sel, bins, _ = ax0.hist(sel, label=label_sel, **plot_options)
    # ax0.set_ylim(1.1, 5000)
    ax0.legend(fontsize=16)
    ax0.set_title(title)

    ratio = [x/y if y != 0 else 0. for x,y in zip(val_sel, val_tot)][:-1]
    xvals = [0.5*(x[0]+x[1]) for x in pairwise(bins)][1:]
    ax1.plot(xvals, ratio, 'o', label='ratio', lw=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('ratio')
    plt.savefig(outname)