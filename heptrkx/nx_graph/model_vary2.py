from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
* Interaction Network:
    * node_block: use_received_edges=False
* edge_encoder:  MLP = [latent_size]*2
* edge_output: only [1]
"""

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import blocks
import sonnet as snt

LATENT_SIZE = 128
NUM_LAYERS = 2

def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS,
                   activation=tf.nn.relu,
                   activate_final=True,
                   use_dropout=True,
                  ),
      snt.LayerNorm()
  ])


class InteractionNetwork(snt.AbstractModule):
  """Implementation of an Interaction Network.

  An interaction networks computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incomming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This model does not update the graph globals, and they are allowed to be
  `None`.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="interaction_network"):
    """Initializes the InteractionNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to `EdgeBlock` to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.EdgeBlock` for details), and the shape of the
        output of this module must match the one of the input nodes, but for the
        first and last axis.
      node_model_fn: A callable that will be passed to `NodeBlock` to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.NodeBlock` for details).
      reducer: Reducer to be used by NodeBlock to aggregate edges. Defaults to
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(InteractionNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn, use_globals=False)
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn,
          use_received_edges=True,
          use_sent_edges=True,
          use_globals=False,
          received_edges_reducer=reducer)

  def _build(self, graph):
    """Connects the InterationNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
        `None`. The features of each node and edge of `graph` must be
        concatenable on the last axis (i.e., the shapes of `graph.nodes` and
        `graph.edges` must match but for their first and last axis).

    Returns:
      An output `graphs.GraphsTuple` with updated edges and nodes.

    Raises:
      ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
        `graph.senders` is `None`.
    """
    return self._edge_block(self._node_block(graph))

class SegmentClassifier(snt.AbstractModule):

    def __init__(self, name="SegmentClassifier"):
        super(SegmentClassifier, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=lambda : snt.nets.MLP([LATENT_SIZE]*2,
                                                activation=tf.nn.relu,
                                                activate_final=True,
                                                use_dropout=True
                                               ),
            #edge_model_fn=make_mlp_model,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block'
        )
        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        self._core = InteractionNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            reducer=tf.unsorted_segment_sum
        )

        # Transforms the outputs into appropriate shapes.
        edge_output_size = 1
        edge_fn =lambda: snt.Sequential([
            snt.nets.MLP([edge_output_size],
                         activation=tf.nn.relu, # default is relu
                         name='edge_output'),
            tf.sigmoid])

        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, None, None)

    def _build(self, input_op, num_processing_steps):
        latent = self._edge_block(self._node_encoder_block(input_op))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

        output_ops.append(self._output_transform(latent))
        return output_ops
