"""
Make doublet GraphNtuple
"""
import numpy as np
import pandas as pd
from graph_nets import utils_tf

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
        # hit_id = hits[mask].hit_id.values
        # n_nodes = hit_id.shape[0]
        # sub_doublets = segments[segments.hit_id_in.isin(hit_id) & segments.hit_id_out.isin(hit_id)]
        hit_id = hits[mask].hit_id.values
        sub_doublets = segments[segments.hit_id_in.isin(hit_id)]
        # extend the hits to include the hits used in the sub-doublets.
        hit_id = hits[mask | hits.hit_id.isin(sub_doublets.hit_id_out.values)].hit_id.values
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
            'edges': sub_doublets.solution.values.astype(np.float32),
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
    def __init__(self, n_eta, n_phi, node_features, edge_features, verbose=False):
        self.n_eta = n_eta
        self.n_phi = n_phi
        self.node_features = node_features
        self.edge_features = edge_features
        self.verbose = verbose
        self.graphs = []
        self.evt_list = []
        self.idx_mgr = None

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
        self.idx_mgr = IndexMgr(len(self.graphs))
        print("DoubletGraphGenerator added {} events, {} graphs".format(n_evts, len(self.graphs)))

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

        input_graphs = utils_tf.data_dicts_to_graphs_tuple(inputs)
        target_graphs = utils_tf.data_dicts_to_graphs_tuple(targets)
        return (input_graphs, target_graphs)