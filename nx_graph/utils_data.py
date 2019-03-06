from graph_nets import utils_np
import networkx as nx

import numpy as np
import os

def data_dict_to_networkx(dd_input, dd_target):
    input_nx = utils_np.data_dict_to_networkx(dd_input)
    target_nx = utils_np.data_dict_to_networkx(dd_target)

    G = nx.Graph()
    for node_index, node_features in input_nx.nodes(data=True):
        G.add_node(node_index, pos=node_features['features'])

    for receiver, sender, features in target_nx.edges(data=True):
        G.add_edge(sender, receiver, solution=features['features'])

    return G


def get_graph(path, evtid, isec=0, n_phi_sections=8, n_eta_sections=2):
    phi_range = (-np.pi, np.pi)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)

    scale = [1000, np.pi/n_phi_sections, 1000]

    file_name = 'event{:09d}_g{:09d}_INPUT.npz'.format(evtid, isec)
    with np.load(os.path.join(path, file_name)) as f:
        input_data_dict = dict(f.items())
    with np.load(os.path.join(path, file_name.replace("INPUT", "TARGET"))) as f:
        target_data_dict = dict(f.items())

    G = data_dict_to_networkx(input_data_dict, target_data_dict)
    # update phi
    phi_min = phi_edges[isec//n_eta_sections]
    phi_max = phi_edges[isec//n_eta_sections+1]
    for node_id, features in G.nodes(data=True):
        new_feature = features['pos']*scale
        new_feature[1] = new_feature[1] + (phi_min + phi_max) / 2
        if new_feature[1] > np.pi:
            new_feature[1] -= 2*np.pi
        if new_feature[1] < -np.pi:
            new_feature[1]+= 2*np.pi

        G.node[node_id].update(pos=new_feature)
    return G



def hitsgraph_to_networkx_graph(G):
    n_nodes, n_edges = G.Ri.shape

    graph = nx.DiGraph()

    ## it is essential to add nodes first
    # the node ID must be [0, N_NODES]
    for i in range(n_nodes):
        graph.add_node(i, pos=G.X[i], solution=0.0)

    for iedge in range(n_edges):
        in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
        out_node_id = G.Ro[:, iedge].nonzero()[0][0]

        # distance as features
        in_node_features  = G.X[in_node_id]
        out_node_features = G.X[out_node_id]
        distance = get_edge_features(in_node_features, out_node_features)
        # add edges, bi-directions
        graph.add_edge(in_node_id, out_node_id, distance=distance, solution=G.y[iedge])
        graph.add_edge(out_node_id, in_node_id, distance=distance, solution=G.y[iedge])
        # add "solution" to nodes
        graph.node[in_node_id].update(solution=G.y[iedge])
        graph.node[out_node_id].update(solution=G.y[iedge])

    # add global features, not used for now
    graph.graph['features'] = np.array([0.])
    return graph
