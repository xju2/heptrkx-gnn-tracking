from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

NUM_LAYERS = 4    # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 64  # Hard-code latent layer sizes for demos.


def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS,
                   activation=tf.nn.selu,
                   activate_final=True),
      snt.LayerNorm()
  ])

class MLPGraphIndependent(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=make_mlp_model,
          node_model_fn=make_mlp_model,
          global_model_fn=None)

  def _build(self, inputs):
    return self._network(inputs)



class SegmentClassifier(snt.AbstractModule):

  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._encoder = MLPGraphIndependent()
    self._core = modules.InteractionNetwork(
        edge_model_fn=make_mlp_model,
        node_model_fn=make_mlp_model,
        reducer=tf.unsorted_segment_sum
    )
    self._decoder = modules.GraphIndependent(
        edge_model_fn=make_mlp_model,
        node_model_fn=None, global_model_fn=None)

    # Transforms the outputs into appropriate shapes.
    edge_output_size = 1
    edge_fn =lambda: snt.Sequential([
        snt.nets.MLP([LATENT_SIZE/2, edge_output_size],
                     activation=tf.nn.selu, # default is relu
                     name='edge_output'),
        tf.sigmoid])

    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(edge_fn, None, None)

  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent

    output_ops = []
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)
        decoded_op = self._decoder(latent)
        output_ops.append(self._output_transform(decoded_op))
    return output_ops
