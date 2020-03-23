"""
Make doublet GraphNtuple
"""
import numpy as np
import pandas as pd
import itertools
import random
from graph_nets import utils_tf
from graph_nets import graphs
import tensorflow as tf


def concat_batch_dim(G):
    """
    G is a GraphNtuple Tensor, with additional dimension for batch-size.
    Concatenate them along the axis for batch
    """
    n_node = tf.reshape(G.n_node, [-1])
    n_edge = tf.reshape(G.n_edge, [-1])
    nodes = tf.reshape(G.nodes, [-1, G.nodes.shape[-1]])
    edges = tf.reshape(G.edges, [-1, G.edges.shape[-1]])
    senders = tf.reshape(G.senders, [-1])
    receivers = tf.reshape(G.receivers, [-1])
    globals_ = tf.reshape(G.globals, [-1, G.globals.shape[-1]])
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def add_batch_dim(G, axis=0):
    """
    G is a GraphNtuple Tensor, without a dimension for batch.
    Add a dimensioin them along the axis for batch
    """
    n_node = tf.expand_dims(G.n_node, axis=0)
    n_edge = tf.expand_dims(G.n_edge, axis=0)
    nodes = tf.expand_dims(G.nodes, axis=0)
    edges = tf.expand_dims(G.edges, axis=0)
    senders = tf.expand_dims(G.senders, axis=0)
    receivers = tf.expand_dims(G.receivers, 0)
    globals_ = tf.expand_dims(G.globals, 0)
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def data_dicts_to_graphs_tuple(input_dd, target_dd, with_batch_dim=True):
    if type(input_dd) is not list:
        input_dd = [input_dd]
    if type(target_dd) is not list:
        target_dd = [target_dd]
        
    input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_dd)
    target_graphs = utils_tf.data_dicts_to_graphs_tuple(target_dd)
    # fill zeros
    input_graphs = utils_tf.set_zero_global_features(input_graphs, 1)
    target_graphs = utils_tf.set_zero_global_features(target_graphs, 1)
    target_graphs = utils_tf.set_zero_node_features(target_graphs, 1)
    
    # expand dims
    if with_batch_dim:
        input_graphs = add_batch_dim(input_graphs)
        target_graphs = add_batch_dim(target_graphs)
    return input_graphs, target_graphs

def specs_from_graphs_tuple(
    graphs_tuple_sample, with_batch_dim=False,
    dynamic_num_graphs=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    description_fn=tf.TensorSpec,
    ):
    graphs_tuple_description_fields = {}
    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]

    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(graphs_tuple_sample, field_name)
        if field_sample is None:
            raise ValueError(
                "The `GraphsTuple` field `{}` was `None`. All fields of the "
                "`GraphsTuple` must be specified to create valid signatures that"
                "work with `tf.function`. This can be achieved with `input_graph = "
                "utils_tf.set_zero_{{node,edge,global}}_features(input_graph, 0)`"
                "to replace None's by empty features in your graph. Alternatively"
                "`None`s can be replaced by empty lists by doing `input_graph = "
                "input_graph.replace({{nodes,edges,globals}}=[]). To ensure "
                "correct execution of the program, it is recommended to restore "
                "the None's once inside of the `tf.function` by doing "
                "`input_graph = input_graph.replace({{nodes,edges,globals}}=None)"
                "".format(field_name))

        shape = list(field_sample.shape)
        dtype = field_sample.dtype

        # If the field is not None but has no field shape (i.e. it is a constant)
        # then we consider this to be a replaced `None`.
        # If dynamic_num_graphs, then all fields have a None first dimension.
        # If dynamic_num_nodes, then the "nodes" field needs None first dimension.
        # If dynamic_num_edges, then the "edges", "senders" and "receivers" need
        # a None first dimension.
        if shape:
            if with_batch_dim:
                shape[1] = None
            elif (dynamic_num_graphs \
                or (dynamic_num_nodes \
                    and field_name == graphs.NODES) \
                or (dynamic_num_edges \
                    and field_name in edge_dim_fields)): shape[0] = None

        graphs_tuple_description_fields[field_name] = description_fn(
            shape=shape, dtype=dtype)

    return graphs.GraphsTuple(**graphs_tuple_description_fields)


def dtype_shape_from_graphs_tuple(input_graph, with_batch_dim=False, debug=False):
    graphs_tuple_dtype = {}
    graphs_tuple_shape = {}

    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]
    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(input_graph, field_name)
        shape = list(field_sample.shape)
        dtype = field_sample.dtype
        # print(field_name, shape, dtype)

        if shape:
            if with_batch_dim:
                shape[1] = None
            else:
                if field_name == graphs.NODES or field_name in edge_dim_fields:
                    shape[0] = None

        graphs_tuple_dtype[field_name] = dtype
        graphs_tuple_shape[field_name] = tf.TensorShape(shape)
        if debug:
            print(field_name, shape, dtype)
    
    return graphs.GraphsTuple(**graphs_tuple_dtype), graphs.GraphsTuple(**graphs_tuple_shape)

# TODO: use one-hot-encoding to add layer info for nodes,
# attach the flattened encoding to node features
def make_graph_ntuples(hits, segments, n_eta, n_phi,
                    node_features=['r', 'phi', 'z'],
                    edge_features=['deta', 'dphi'],
                    dphi=0.0, deta=0.0, verbose=False):
    phi_range = (-np.pi, np.pi)
    eta_range = (-5, 5)
    phi_edges = np.linspace(*phi_range, num=n_phi+1)
    eta_edges = np.linspace(*eta_range, num=n_eta+1)

    n_graphs = n_eta * n_phi
    if verbose:
        print("{} graphs".format(n_graphs))

    def make_subgraph(mask):
        hit_id = hits[mask].hit_id.values
        sub_doublets = segments[segments.hit_id_in.isin(hit_id) & segments.hit_id_out.isin(hit_id)]

        # sub_doublets = segments[segments.hit_id_in.isin(hit_id)]
        # # extend the hits to include the hits used in the sub-doublets.
        # hit_id = hits[mask | hits.hit_id.isin(sub_doublets.hit_id_out.values)].hit_id.values


        n_nodes = hit_id.shape[0]
        n_edges = sub_doublets.shape[0]
        nodes = hits[mask][node_features].values.astype(np.float64)
        edges = sub_doublets[edge_features].values
        # print(nodes.dtype)

        hits_id_dict = {}
        for idx in range(n_nodes):
            hits_id_dict[hit_id[idx]] = idx

        senders = []
        receivers = []
        in_hit_ids = sub_doublets.hit_id_in.values
        out_hit_ids = sub_doublets.hit_id_out.values
        for idx in range(n_edges):
            senders.append( hits_id_dict[in_hit_ids[idx]] )
            receivers.append( hits_id_dict[out_hit_ids[idx]] )
        if verbose:
            print("\t{} nodes and {} edges".format(n_nodes, n_edges))
        senders = np.array(senders)
        receivers = np.array(receivers)
        return ({
            "n_node": n_nodes,
            'n_edge': n_edges,
            'nodes': nodes,
            'edges': edges,
            'senders': senders,
            'receivers': receivers,
        }, {
            "n_node": n_nodes,
            'n_edge': n_edges,
            # 'nodes': None,
            'edges': np.expand_dims(sub_doublets.solution.values.astype(np.float32), axis=1),
            'senders': senders,
            'receivers': receivers,
        }, 
        )

    all_graphs = []
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        phi_max += dphi
        phi_min -= dphi
        phi_mask = (hits.phi > phi_min) & (hits.phi < phi_max)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            eta_min -= deta
            eta_max += deta
            eta_mask = (hits.eta > eta_min) & (hits.eta < eta_max)
            all_graphs.append(make_subgraph(eta_mask & phi_mask))
    tot_nodes = sum([x['n_node'] for x, _ in all_graphs])
    tot_edges = sum([x['n_edge'] for x, _ in all_graphs])
    if verbose:
        print("\t{} nodes and {} edges".format(tot_nodes, tot_edges))
    return all_graphs

class IndexMgr:
    def __init__(self, n_total, training_frac=0.8):
        self.max_tr = int(n_total*training_frac)
        self.total = n_total
        self.n_test = n_total - self.max_tr
        self.tr_idx = 0
        self.te_idx = self.max_tr

    def next(self, is_training=False):
        if is_training:
            self.tr_idx += 1
            if self.tr_idx > self.max_tr:
                self.tr_idx = 0
            return self.tr_idx
        else:
            self.te_idx += 1
            if self.te_idx > self.total:
                self.te_idx = self.max_tr
            return self.te_idx


class DoubletGraphGenerator:
    def __init__(self, n_eta, n_phi, node_features, edge_features, with_batch_dim=True, verbose=False):
        self.n_eta = n_eta
        self.n_phi = n_phi
        self.node_features = node_features
        self.edge_features = edge_features
        self.verbose = verbose
        self.graphs = []
        self.evt_list = []
        self.idx_mgr = None
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_batch_dim = with_batch_dim

    def add_file(self, hit_file, doublet_file):
        with pd.HDFStore(hit_file, 'r') as hit_store:
            n_evts = len(list(hit_store.keys()))
            with pd.HDFStore(doublet_file, 'r') as doublet_store:
                n_doublet_keys = len(list(doublet_store.keys()))
                for key in hit_store.keys(): # loop over events
                    doublets = []
                    try:
                        for ipair in range(9):
                            pair_key = key+'/pair{}'.format(ipair)
                            doublets.append(doublet_store[pair_key])
                    except KeyError:
                        continue
                    hit = hit_store[key]
                    doublets = pd.concat(doublets)
                    all_graphs = make_graph_ntuples(
                                        hit, doublets,
                                        self.n_eta, self.n_phi,
                                        node_features=self.node_features,
                                        edge_features=self.edge_features,
                                        verbose=self.verbose)
                    self.graphs += all_graphs
                    self.evt_list.append(key)

        self.tot_data = len(self.graphs)
        self.idx_mgr = IndexMgr(self.tot_data)
        print("DoubletGraphGenerator added {} events, Total {} graphs".format(n_evts, len(self.graphs)))

    def _get_signature(self):
        if self.input_dtype and self.target_dtype:
            return
        ex_input, ex_target = self.create_graph(num_graphs=1)
        self.input_dtype, self.input_shape = dtype_shape_from_graphs_tuple(
            ex_input, self.with_batch_dim, self.verbose)
        self.target_dtype, self.target_shape = dtype_shape_from_graphs_tuple(
            ex_target, self.with_batch_dim, self.verbose)
        
    def _graph_generator(self, is_training=True): # one graph a dataset
        min_idx, max_idx = 0, int(self.tot_data * 0.8)

        if not is_training:
            min_idx, max_idx = int(self.tot_data*0.8), self.tot_data-1

        for idx in range(min_idx, max_idx):
            yield data_dicts_to_graphs_tuple(self.graphs[idx][0], self.graphs[idx][1], self.with_batch_dim)


    def create_dataset(self, is_training=True):
        self._get_signature()
        dataset = tf.data.Dataset.from_generator(
            self._graph_generator,
            output_types=(self.input_dtype, self.target_dtype),
            output_shapes=(self.input_shape, self.target_shape),
            args=(is_training,)
        )
        return dataset

    # FIXME: 
    # everytime check if one event is completely used (used all subgraphs)
    # shuffle the events, but feed the subgraphs in order
    def create_graph(self, num_graphs, is_training=True):
        if not self.idx_mgr:
            raise ValueError("No Doublet Graph is created")

        inputs = []
        targets = []
        for _ in range(num_graphs):
            idx = self.idx_mgr.next(is_training)
            input_dd, target_dd =  self.graphs[idx]
            inputs.append(input_dd)
            targets.append(target_dd)

        return data_dicts_to_graphs_tuple(inputs, targets, self.with_batch_dim)