#!/usr/bin/env python

import numpy as np
import sklearn.metrics

import tensorflow as tf

from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import utils_np


def create_feed_dict(batch_size, input_ph, target_ph):
    inputs, targets = generate_input_target(batch_size)
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}

    return feed_dict


def create_loss_ops(target_op, output_ops):
    # only use edges
    loss_ops = [
        tf.losses.sigmoid_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def computer_matrics(target, output):
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)

    test_target = []
    test_pred = []
    for td, od in zip(tdds, odds):
        test_target.append(td['edges'])
        test_pred.append(od['edges'])

    test_target = np.concatenate(test_target, axis=0)
    test_pred   = np.concatenate(test_pred,   axis=0)

    thresh = 0.5
    y_pred, y_true = (test_pred > thresh), (test_target > thresh)
    return sklearn.metrics.precision_score(y_true, y_pred), sklearn.metrics.recall_score(y_true, y_pred)


if __name__ == "__main__":
    from nx_graph.model import SegmentClassifier
    import time

    from nx_graph.prepare import inputs_generator
    base_dir = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_001/event00000{}_g{:03d}.npz'
    isec = 0
    generate_input_target = inputs_generator(base_dir, isec)

    tf.reset_default_graph()

    model = SegmentClassifier()
    batch_size = n_graphs = 2
    num_training_iterations = 10000
    num_processing_steps_tr = 4  ## level of message-passing

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_networkxs(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)

    # Training loss.
    loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

    # Optimizer
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

    #@title Reset session  { form-width: "30%" }

    # This cell resets the Tensorflow session, but keeps the same computational
    # graph.

    try:
      sess.close()
    except NameError:
      pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []

    # How much time between logging and printing the current results.
    log_every_seconds = 20

    print("# (iteration number), T (elapsed seconds), "
          "Ltr (training loss), "
          "Precision, "
          "Recall")
    start_time = time.time()
    last_log_time = start_time
    for iteration in range(last_iteration, num_training_iterations):
      last_iteration = iteration
      feed_dict = create_feed_dict(batch_size, input_ph, target_ph)
      train_values = sess.run({
          "step": step_op,
          "target": target_ph,
          "loss": loss_op_tr,
          "outputs": output_ops_tr
      }, feed_dict=feed_dict)
      the_time = time.time()
      elapsed_since_last_log = the_time - last_log_time

      if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time
        feed_dict = create_feed_dict(batch_size, input_ph, target_ph)
        test_values = sess.run({
            "target": target_ph,
            "loss": loss_op_tr,
            "outputs": output_ops_tr
        }, feed_dict=feed_dict)
        correct_tr, solved_tr = computer_matrics(
            test_values["target"], test_values["outputs"][-1])
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        corrects_tr.append(correct_tr)
        solveds_tr.append(solved_tr)
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Precision {:.4f}, Recall"
              " {:.4f}".format(
                  iteration, elapsed, train_values["loss"], test_values["loss"],
                  correct_tr, solved_tr))
