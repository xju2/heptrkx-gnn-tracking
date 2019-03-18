import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import utils_np

import yaml
import os
import numpy as np

from nx_graph.prepare import inputs_generator
from nx_graph import get_model
from nx_graph import utils_data

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)


ckpt_name = 'checkpoint_{:05d}.ckpt'


def create_evaluator(config_name, iteration, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    config = load_config(config_name)
    config_tr = config['train']

    log_every_seconds       = config_tr['time_lapse']
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
        output_nxs = utils_np.graphs_tuple_to_networkxs(output)
        input_dds  = utils_np.graphs_tuple_to_data_dicts(input_graphs)
        target_dds = utils_np.graphs_tuple_to_data_dicts(target_graphs)
        graphs = []
        ig = 0
        for input_dd,target_dd in zip(input_dds, target_dds):
            graph = utils_data.data_dict_to_networkx(
                input_dd, target_dd,
                use_digraph=use_digraph, bidirection=bidirection)

            ## update edge features with TF output
            for edge in graph.edges():
                graph.edges[edge]['predict'] = output_nxs[ig].edges[edge+(0,)]['features']
            graphs.append(graph)
            ig += 1

        return graphs

    return evaluator
