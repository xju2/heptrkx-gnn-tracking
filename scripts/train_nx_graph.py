#!/usr/bin/env python
"""
Training GNN
"""

import tensorflow as tf

import os
import sys
import argparse
import glob
import re
import time
import random
import functools

import numpy as np
import sklearn.metrics


from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from heptrkx.dataset import graph
from heptrkx.nx_graph import get_model
from heptrkx.utils import load_yaml
# ckpt_name = 'checkpoint_{:05d}.ckpt'
ckpt_name = 'checkpoint'

prog_name = os.path.basename(sys.argv[0])

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
        test_target.append(np.squeeze(td['edges']))
        test_pred.append(np.squeeze(od['edges']))

    test_target = np.concatenate(test_target, axis=0)
    test_pred   = np.concatenate(test_pred,   axis=0)
    return test_pred, test_target


def compute_matrics(target, output, thresh=0.5):
    test_pred, test_target = eval_output(target, output)
    y_pred, y_true = (test_pred > thresh), (test_target > thresh)
    return sklearn.metrics.precision_score(y_true, y_pred), sklearn.metrics.recall_score(y_true, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('config', help='configuration file')
    args = parser.parse_args()

    all_config = load_yaml(args.config)
    config = all_config['gnn_training']


    # add ops to save and restore all the variables
    prod_name = config['prod_name']
    output_dir = os.path.join(config['output_dir'], prod_name)
    print("[{}] save models at {}".format(prog_name, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # files = glob.glob(output_dir+"/*.ckpt.meta")
    # last_iteration = 0 if len(files) < 1 else max([
    #     int(re.search('checkpoint_([0-9]*).ckpt.meta', os.path.basename(x)).group(1))
    #     for x in files
    # ])
    # print("[{}] last iteration: {}".format(prog_name, last_iteration))

    config_tr = config['parameters']
    log_every_seconds       = config_tr['time_lapse']
    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_training_iterations = config_tr['iterations']
    iter_per_job            = 2500 if 'iter_per_job' not in config_tr else config_tr['iter_per_job']
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    print("Maximum iterations per job: {}".format(iter_per_job))

    learning_rate = config_tr['learning_rate']
    optimizer = snt.optimizers.Adam(learning_rate)
    model = get_model(config['model_name'])
 
    # prepare graphs
    with_batch_dim = False
    print("Node features: ", config['node_features'])
    print("Edge features: ", config['edge_features'])
    doublet_graphs = graph.DoubletGraphGenerator(
        config['n_eta'], config['n_phi'],
        config['node_features'], config['edge_features'],
        with_batch_dim=with_batch_dim
        )
    for hit_file, doublet_file in zip(config['hit_files'], config['doublet_files']):
        doublet_graphs.add_file(hit_file, doublet_file)

    @tf.function
    def get_data():
        in_graphs, out_graphs = doublet_graphs.create_graph(batch_size, is_training=True)        
        return in_graphs, out_graphs

    @tf.function
    def get_test_data():
        in_graphs, out_graphs = doublet_graphs.create_graph(batch_size, is_training=False)
        return in_graphs, out_graphs

    # inputs, targets = doublet_graphs.create_graph(batch_size)
    inputs, targets = get_data()
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim)
    )

    # training loss
    if config_tr['real_weight']:
        real_weight = config_tr['real_weight']
        fake_weight = config_tr['fake_weight']
    else:
        real_weight = fake_weight = 1.0

    def create_loss_ops(target_op, output_ops):
        weights = target_op.edges * real_weight + (1 - target_op.edges) * fake_weight
        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.edges, output_op.edges, weights=weights)
            for output_op in output_ops
        ]
        return tf.stack(loss_ops)

    @functools.partial(tf.function, input_signature=input_signature)
    def update_step(inputs_tr, targets_tr):
        print("Tracing update_step")
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr, num_processing_steps_tr)
            loss_ops_tr = create_loss_ops(targets_tr, outputs_tr)
            loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

        gradients = tape.gradient(loss_op_tr, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply(gradients, model.trainable_variables)
        return outputs_tr, loss_op_tr

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=5)
    if os.path.exists(os.path.join(output_dir, ckpt_name)):
        print("Loading latest checkpoint")
        status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []

    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), T (elapsed seconds), Ltr (training loss), Precision, Recall\n"
    log_name = os.path.join(output_dir, config['log_name'])
    with open(log_name, 'a') as f:
        f.write(out_str)

    start_time = time.time()
    last_log_time = start_time
    ## loop over iterations, each iteration generating a batch of data for training
    iruns = 0
    print("# (iteration number), TD (get graph), TR (TF run)")
    last_iteration = 0
    for iteration in range(last_iteration, num_training_iterations):
        if iruns > iter_per_job:
            print("runs larger than {} iterations per job, stop".format(iter_per_job))
            break
        else: iruns += 1
        last_iteration = iteration

        inputs_tr, targets_tr = get_data()
        outputs_tr, loss_tr = update_step(inputs_tr, targets_tr)


        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        if elapsed_since_last_log > log_every_seconds:
            # save a checkpoint
            ckpt_manager.save()

            last_log_time = the_time
            inputs_te, targets_te = get_test_data()
            outputs_te, loss_te = update_step(inputs_te, targets_te)
            correct_tr, solved_tr = compute_matrics(targets_te, outputs_te[-1])

            elapsed = time.time() - start_time
            logged_iterations.append(iteration)
            out_str = "# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Precision {:.4f}, Recall {:.4f}".format(
                iteration, elapsed, loss_tr, loss_te,
                correct_tr, solved_tr)
            print(out_str)