"""
convert hitgraphs to network-x and prepare graphs for graph-nets
"""
import numpy as np
from datasets.graph import load_graph
import networkx as nx
import os
import glob

def get_edge_features(in_node, out_node):
    # input are the features of incoming and outgoing nodes
    # they are ordered as [r, phi, z]
    in_r, in_phi, _   = in_node
    out_r, out_phi, _ = out_node
    in_x = in_r * np.cos(in_phi)
    in_y = in_r * np.sin(in_phi)
    out_x = out_r * np.cos(out_phi)
    out_y = out_r * np.sin(out_phi)
    return np.sqrt((in_x - out_x)**2 + (in_y - out_y)**2)


def hitsgraph_to_networkx_graph(G):
    n_nodes, n_edges = G.Ri.shape

    graph = nx.DiGraph()

    ## add nodes
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


def graph_to_input_target(graph):
    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos",)
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields)
        )
        target_graph.add_node(
            node_index, features=create_feature(node_feature, target_node_fields)
        )

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields)
        )
        target_graph.add_edge(
            sender, receiver, features=create_feature(features, target_edge_fields)
        )

    input_graph.graph['features'] = input_graph.graph['features'] = np.array([0.0])
    return input_graph, target_graph


def inputs_generator(base_dir_, isec_=0):
    file_patten = base_dir.format(1000, 0).replace('1000', '*')
    max_evt_id = max([
        int(re.search('event00000([0-9]*)_g000.npz', os.path.basename(x)).group(1))
        for x in glob.glob(file_patten)
    ])
    base_dir = base_dir_
    isec     = isec_
    global _evt_id_
    _evt_id_ = 1000
    def generate_input_target(n_graphs):
        global _evt_id_
        input_graphs = []
        target_graphs = []
        igraphs = 0
        while igraphs < n_graphs:
            file_name = base_dir.format(_evt_id_, isec)
            while not os.path.exits(file_name):
                _evt_id_ += 1
                if _evt_id_ > max_evt_id:
                    _evt_id_ = 1000
                file_name = base_dir.format(_evt_id_, isec)

            graph = hitsgraph_to_networkx_graph(load_graph(file_name))
            input_graph, output_graph = graph_to_input_target(graph)
            input_graphs.append(input_graph)
            target_graphs.append(output_graph)
            _evt_id_ += i

        return input_graphs, target_graphs

    return generate_input_target
