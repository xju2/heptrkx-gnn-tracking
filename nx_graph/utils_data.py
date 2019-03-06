from graph_nets import utils_np
import networkx as nx

import numpy as np
import os


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi


def get_edge_features(in_node, out_node):
    # input are the features of incoming and outgoing nodes
    # they are ordered as [r, phi, z]
    in_r, in_phi, in_z    = in_node
    out_r, out_phi, out_z = out_node

    in_r3 = np.sqrt(in_r**2 + in_z**2)
    out_r3 = np.sqrt(out_r**2 + out_z**2)

    in_theta = np.arccos(in_z/in_r3)
    in_eta = -np.log(np.tan(in_theta/2.0))
    out_theta = np.arccos(out_z/out_r3)
    out_eta = -np.log(np.tan(out_theta/2.0))
    deta = out_eta - in_eta
    dphi = calc_dphi(out_phi, in_phi)
    dR = np.sqrt(deta**2 + dphi**2)
    dZ = in_z - out_z
    return np.array([deta, dphi, dR, dZ])


def data_dict_to_networkx(dd_input, dd_target, use_digraph=True):
    input_nx  = utils_np.data_dict_to_networkx(dd_input)
    target_nx = utils_np.data_dict_to_networkx(dd_target)

    G = nx.DiGraph() if use_digraph else nx.Graph()
    for node_index, node_features in input_nx.nodes(data=True):
        G.add_node(node_index, pos=node_features['features'])

    for sender, receiver, features in target_nx.edges(data=True):
        G.add_edge(sender, receiver, solution=features['features'])
        if use_digraph:
            G.add_edge(receiver, sender, solution=features['features'])

    return G


def get_graph_from_saved_data_dict(path, evtid, isec=0, n_phi_sections=8, n_eta_sections=2,
              use_digraph=True, do_correction=False):

    file_name = 'event{:09d}_g{:09d}_INPUT.npz'.format(evtid, isec)
    with np.load(os.path.join(path, file_name)) as f:
        input_data_dict = dict(f.items())
    with np.load(os.path.join(path, file_name.replace("INPUT", "TARGET"))) as f:
        target_data_dict = dict(f.items())

    G = data_dict_to_networkx(input_data_dict, target_data_dict, use_digraph)
    if not do_correction:
        return G

    phi_range = (-np.pi, np.pi)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    scale = [1000, np.pi/n_phi_sections, 1000]
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


def hitsgraph_to_networkx_graph(G, use_digraph=True):
    n_nodes, n_edges = G.Ri.shape

    graph = nx.DiGraph()

    ## it is essential to add nodes first
    # the node ID must be [0, N_NODES]
    for i in range(n_nodes):
        graph.add_node(i, pos=G.X[i], solution=0.0)

    for iedge in range(n_edges):
        """
        In_node:  node is a receiver, hits at outer-most layers can only be In-node
        Out-node: node is a sender, so hits in inner-most layer can only be Out-node
        """
        in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
        out_node_id = G.Ro[:, iedge].nonzero()[0][0]

        # distance as features
        in_node_features  = G.X[in_node_id]
        out_node_features = G.X[out_node_id]
        distance = get_edge_features(in_node_features, out_node_features)
        # add edges, bi-directions
        # connection of inner to outer
        graph.add_edge(out_node_id, in_node_id, distance=distance, solution=G.y[iedge])
        # connection of outer to inner
        if use_digraph:
            graph.add_edge(in_node_id, out_node_id, distance=distance, solution=G.y[iedge])
        # add "solution" to nodes
        graph.node[in_node_id].update(solution=G.y[iedge])
        graph.node[out_node_id].update(solution=G.y[iedge])

    # add global features, not used for now
    graph.graph['features'] = np.array([0.])
    return graph


def networkx_graph_to_hitsgraph(G, is_digraph=True):
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())//2 if is_digraph else len(G.edges())
    n_features = len(G.node[0]['pos'])

    X = np.zeros((n_nodes, n_features), dtype=np.float32)
    Ri = np.zeros((n_nodes, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_nodes, n_edges), dtype=np.uint8)

    for node,features in G.nodes(data=True):
        X[node, :] = features['pos']

    ## build relations
    segments = []
    y = []
    for n, nbrsdict in G.adjacency():
        for nbr, eattr in nbrsdict.items():
            ## as hitsgraph is a directed graph from inner-most to outer-most
            ## so assume sender < reciver;
            if n > nbr and is_digraph:
                continue
            segments.append((n, nbr))
            y.append(int(eattr['solution'][0]))

    if len(y) != n_edges:
        print(len(y),"not equals to # of edges", n_edges)
    segments = np.array(segments)
    Ro[segments[:, 0], np.arange(n_edges)] = 1
    Ri[segments[:, 1], np.arange(n_edges)] = 1
    y = np.array(y, dtype=np.float32)
    return (X, Ri, Ro, y)


def is_diff_networkx(G1, G2):
    """
    G1,G2, networkx graphs
    Return True if they are different, False otherwise
    note that edge features are not checked!
    """
    # check node features first
    GRAPH_NX_FEATURES_KEY = 'pos'
    node_id1 = np.array([
        x[1][GRAPH_NX_FEATURES_KEY]
        for x in G1.nodes(data=True)
        if x[1][GRAPH_NX_FEATURES_KEY] is not None])
    node_id2 = np.array([
        x[1][GRAPH_NX_FEATURES_KEY]
        for x in G2.nodes(data=True)
        if x[1][GRAPH_NX_FEATURES_KEY] is not None])

    # check edges
    diff = np.any(node_id1 != node_id2)
    for sender, receiver, _ in G1.edges(data=True):
        try:
            _ = G2.edges[(sender, receiver)]
        except KeyError:
            diff = True
            break
    return diff
