import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import utils_np

import yaml
import os
import numpy as np

from ..nx_graph.prepare import inputs_generator
from ..nx_graph import get_model, utils_data, utils_train, utils_io
from ..nx_graph.utils_io import ckpt_name
from .. import load_yaml


def create_evaluator(config_name, iteration, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    config = load_yaml(config_name)
    config_tr = config['train']

    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    if input_ckpt is None:
        input_ckpt = os.path.join(config['output_dir'], prod_name)


    # generate inputs
    generate_input_target = inputs_generator(config['data']['output_nxgraph_dir'], n_train_fraction=0.8)

    # build TF graph
    tf.reset_default_graph()
    model = get_model(config['model']['name'])

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)
    try:
        sess.close()
    except NameError:
        pass

    sess = tf.Session()
    saver = tf.train.Saver()
    if iteration < 0:
        saver.restore(sess, tf.train.latest_checkpoint(input_ckpt))
    else:
        saver.restore(sess, os.path.join(input_ckpt, ckpt_name.format(iteration)))

    def evaluator(input_graphs, target_graphs, use_digraph=False, bidirection=False):
        """
        input is graph tuples, sizes should match batch_size
        """
        feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
        predictions = sess.run({
            "outputs": output_ops_tr,
            "target": target_ph
        }, feed_dict=feed_dict)
        output = predictions['outputs'][-1]

        return utils_data.predicted_graphs_to_nxs(
            output, input_graphs, target_graphs,
            use_digraph=use_digraph,
            bidirection=bidirection)

    return evaluator
