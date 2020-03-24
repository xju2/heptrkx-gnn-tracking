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
    global_batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_training_iterations = config_tr['iterations']
    iter_per_job            = 2500 if 'iter_per_job' not in config_tr else config_tr['iter_per_job']
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    print("Maximum iterations per job: {}".format(iter_per_job))

    # model and optimizers
    learning_rate = config_tr['learning_rate']
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    physical_cpus = tf.config.experimental.list_physical_devices("CPU")
    print(physical_cpus)
    n_gpus = len(physical_gpus)
    with_batch_dim = True
    with_pad = True
    if n_gpus > 1:
        print("Useing SNT Replicator with {} workers".format(n_gpus))
        strategy = snt.distribute.Replicator(['/device:GPU:{}'.format(i) for i in range(n_gpus)],\
            tf.distribute.ReductionToOneDevice("GPU:0"))
    elif n_gpus > 0:
        strategy = tf.distribute.OneDeviceStrategy("/device:GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/device:CPU:0")

    with strategy.scope():
        optimizer = snt.optimizers.Adam(learning_rate)
        model = get_model(config['model_name'])

    # prepare graphs
    print("Node features: ", config['node_features'])
    print("Edge features: ", config['edge_features'])
    doublet_graphs = graph.DoubletGraphGenerator(
        config['n_eta'], config['n_phi'],
        config['node_features'], config['edge_features'], 
        with_batch_dim=with_batch_dim,
        with_pad=with_pad
        )
    for hit_file, doublet_file in zip(config['hit_files'], config['doublet_files']):
        doublet_graphs.add_file(hit_file, doublet_file)

    training_dataset = doublet_graphs.create_dataset(is_training=True)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    training_dataset = training_dataset.cache()

    # distributed dataset
    dist_training_dataset = strategy.experimental_distribute_dataset(training_dataset)

    testing_dataset = doublet_graphs.create_dataset(is_training=False)


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

    @tf.function
    def train_step(inputs_tr, targets_tr):
        print("Tracing train_step")
        
        def update_step(inputs_tr, targets_tr):
            print("Tracing update_step")
            inputs_tr = graph.concat_batch_dim(inputs_tr)
            targets_tr = graph.concat_batch_dim(targets_tr)
            with tf.GradientTape() as tape:
                outputs_tr = model(inputs_tr, num_processing_steps_tr)
                loss_ops_tr = create_loss_ops(targets_tr, outputs_tr)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, model.trainable_variables)
            # aggregate the gradients from the full batch.
            # this is not there for mirror strategy
            replica_ctx = tf.distribute.get_replica_context()
            gradients = replica_ctx.all_reduce("mean", gradients)

            optimizer.apply(gradients, model.trainable_variables)
            return loss_op_tr


        per_example_losses = strategy.experimental_run_v2(update_step, args=(inputs_tr,targets_tr))
        mean_loss = strategy.reduce("sum", per_example_losses, axis=None)
        return mean_loss

    def train_epoch(dataset):
        total_loss = 0.
        num_batches = 0
        for inputs in dataset:
            print("Step {}".format(num_batches))
            input_tr, target_tr = inputs
            total_loss += train_step(input_tr, target_tr).numpy()
            num_batches += 1
        return total_loss/num_batches

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=5)
    if os.path.exists(os.path.join(output_dir, ckpt_name)):
        print("Loading latest checkpoint")
        status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    n_epochs = config_tr['epochs']
    for epoch in range(n_epochs):
        print("Training epoch", epoch, "...", end=' ')
        print("Loss :=", train_epoch(dist_training_dataset))
        ckpt_manager.save()