
import numpy as np
import sklearn.metrics


from graph_nets import utils_np

def create_feed_dict(generator, batch_size, input_ph, target_ph, is_trained=True):
    inputs, targets = generator(batch_size, is_trained)
    if isinstance(inputs[0], dict):
        input_graphs  = utils_np.data_dicts_to_graphs_tuple(inputs)
        target_graphs = utils_np.data_dicts_to_graphs_tuple(targets)
    else:
        input_graphs  = utils_np.networkxs_to_graphs_tuple(inputs)
        target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}

    return feed_dict


def create_loss_ops(target_op, output_ops, weights):
    import tensorflow as tf
    # only use edges
    loss_ops = [
        tf.compat.v1.losses.log_loss(target_op.edges, output_op.edges, weights=weights)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    from graph_nets import utils_tf
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]