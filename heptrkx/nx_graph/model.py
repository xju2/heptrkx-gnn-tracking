from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

NUM_LAYERS = 4    # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 256 # Hard-code latent layer sizes for demos.


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


class SegmentClassifier(snt.Module):
  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._encoder = MLPGraphIndependent()
    self._core = modules.InteractionNetwork(
      make_mlp_model, make_mlp_model, reducer=tf.math.unsorted_segment_sum
    )

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