import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets import _base
from graph_nets import modules
import sonnet as snt

class EdgeBlock(_base.AbstractModule):
    def __init__(self, edge_model_fn, name='dist_edge_block'):    
        super(EdgeBlock, self).__init__(name=name)
        with self._enter_variable_scope():
            self._edge_model = edge_model_fn()
    
    def _build(self, graph):
        agg_receiver_nodes_features = blocks.broadcast_receiver_nodes_to_edges(graph)
        agg_sender_nodes_features = blocks.broadcast_sender_nodes_to_edges(graph)

        # aggreate across replicas

        replica_ctx = tf.distribute.get_replica_context()
        agg_receiver_nodes_features = replica_ctx.all_reduce("sum", agg_receiver_nodes_features)
        agg_sender_nodes_features = replica_ctx.all_reduce("sum", agg_sender_nodes_features)

        edges_to_collect = [graph.edges, agg_receiver_nodes_features, agg_sender_nodes_features]
        collected_edges = tf.concat(edges_to_collect, axis=-1)
        updated_edges = self._edge_model(collected_edges)
        return graph.replace(edges=updated_edges)


class NodeBlock(_base.AbstractModule):
    def __init__(self, node_model_fn,
        received_edges_reducer=tf.math.unsorted_segment_sum,
        sent_edges_reducer=tf.math.unsorted_segment_sum,
        name='dist_node_block'):
        super(NodeBlock, self).__init__(name=name)
        with self._enter_variable_scope():
            self._received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(received_edges_reducer)
            self._sent_edges_aggregator = blocks.SentEdgesToNodesAggregator(sent_edges_reducer)
            self._node_model = node_model_fn()

    def _build(self, graph):

        received_edges_features = self._received_edges_aggregator(graph)
        sent_edges_features = self._sent_edges_aggregator(graph)

        # aggreate all edge info across replicas info to the node in question
        # the aggreated information then used to tune the node network

        replica_ctx = tf.distribute.get_replica_context()
        received_edges_features = replica_ctx.all_reduce("sum", received_edges_features)
        sent_edges_features = replica_ctx.all_reduce("sum", sent_edges_features)

        nodes_to_collect = [graph.nodes, received_edges_features, sent_edges_features]

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        return graph.replace(nodes=updated_nodes)


NUM_LAYERS = 2    # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 128 # Hard-code latent layer sizes for demos.


def make_mlp_model():
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
        A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
        snt.nets.MLP([128, 64],
                    activation=tf.nn.relu,
                    activate_final=True),
        # snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


class MLPGraphIndependent(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=None)

    def __call__(self, inputs):
        return self._network(inputs)


class DistInteractionNetwork(_base.AbstractModule):
    def __init__(self, edge_model_fn, node_model_fn,
                reducer=tf.math.unsorted_segment_sum,
                name='DistInteractionNetwork'):
        super(DistInteractionNetwork, self).__init__(name=name)
        self._edge_block = EdgeBlock(edge_model_fn=edge_model_fn)
        self._node_block = NodeBlock(node_model_fn=node_model_fn)

    def _build(self, graph):
        return self._node_block(self._edge_block(graph))


class SegmentClassifier(snt.Module):
  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._encoder = MLPGraphIndependent()
    self._core = DistInteractionNetwork(make_mlp_model, make_mlp_model)

    self._decoder = modules.GraphIndependent(
        edge_model_fn=make_mlp_model,
        node_model_fn=None, global_model_fn=None)

    # Transforms the outputs into appropriate shapes.
    edge_output_size = 1
    # edge_fn = lambda: snt.Linear(edge_output_size, name='edge_output')
    edge_fn =lambda: snt.Sequential([
        snt.nets.MLP([edge_output_size],
                     activation=tf.nn.relu, # default is relu
                     name='edge_output'),
        tf.sigmoid])
    self._output_transform = modules.GraphIndependent(edge_fn, None, None)

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent

    output_ops = []
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)
        decoded_op = self._decoder(latent)
        output = self._output_transform(decoded_op)
        output_ops.append(output)
    return output_ops