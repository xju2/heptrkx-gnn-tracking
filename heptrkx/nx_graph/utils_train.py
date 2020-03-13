
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


def eval_output(target, output):
    """
    target, output are graph-tuple from TF-GNN,
    each of them contains N=batch-size graphs
    """
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)

    test_target = []
    test_pred = []
    for td, od in zip(tdds, odds):
        test_target.append(td['edges'])
        test_pred.append(od['edges'])

    test_target = np.concatenate(test_target, axis=0)
    test_pred   = np.concatenate(test_pred,   axis=0)
    return test_pred, test_target


def compute_matrics(target, output, thresh=0.5):
    test_pred, test_target = eval_output(target, output)
    y_pred, y_true = (test_pred > thresh), (test_target > thresh)
    return sklearn.metrics.precision_score(y_true, y_pred), sklearn.metrics.recall_score(y_true, y_pred)