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
        test_target.append(td['edges'])
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
        with_batch_dim=True
        )
    for hit_file, doublet_file in zip(config['hit_files'], config['doublet_files']):
        doublet_graphs.add_file(hit_file, doublet_file)


    # to get signature
    inputs, targets = doublet_graphs.create_graph(1)
    inputs = graph.add_batch_dim(inputs)
    targets = graph.add_batch_dim(targets)
    input_signature = (
        graph.specs_from_graphs_tuple(inputs),
        graph.specs_from_graphs_tuple(targets)
    )

    # this prompt an error
    # Cannot batch tensors with different shapes in component 0.
    # First element had shape [3442,3] and element 1 had shape [3604,3]
    # training_dataset = training_dataset.batch(batch_size) // does not work, batch size moves to data generator
    # training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # training_dataset = training_dataset.cache()

    # distributed dataset
    # dist_training_dataset = strategy.experimental_distribute_dataset(training_dataset)
    def training_data_fn(input_context):
        dataset = doublet_graphs.create_dataset()
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        print('batch size per replica: {}, and {} pipelines, and {} pipeline id'.format(
            batch_size, input_context.num_input_pipelines, input_context.input_pipeline_id
        ))
        # d = dataset.prefetch(tf.data.experimental.AUTOTUNE).take(batch_size).cache().repeat()
        return dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

    dist_training_dataset = strategy.experimental_distribute_datasets_from_function(training_data_fn)

    testing_dataset = doublet_graphs.create_dataset(is_training=False)


    # training loss
    if config_tr['real_weight']:
        real_weight = config_tr['real_weight']
        fake_weight = config_tr['fake_weight']
    else:
        real_weight = fake_weight = 1.0

    def create_loss_ops(target_op, output_ops):
        # only use edges
        weights = target_op * real_weight + (1 - target_op) * fake_weight
        # print(output_ops[0].edges.shape)
        # print(target_op.edges.shape)
        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op, output_op, weights=weights)
            for output_op in output_ops
        ]
        return loss_ops



    @functools.partial(tf.function, input_signature=(input_signature,))
    def train_step(dist_inputs):
        print("print once")

        def update_step(inputs):
            inputs_tr, targets_tr = inputs
            # the distributed dataset introduced an dimension with the size of batch_size
            # I have to reshape that.
            # Need to be a bit careful!
            inputs_tr = graph.concat_batch_dim(inputs_tr)
            targets_tr = graph.concat_batch_dim(targets_tr)
            # print(inputs_tr)
            # print(targets_tr)

            with tf.GradientTape() as tape:
                outputs_tr = model(inputs_tr, num_processing_steps_tr)
                outputs_tr = [tf.reshape(x.edges, [-1]) for x in outputs_tr]
                targets_tr = tf.reshape(targets_tr.edges, [-1])
                # print(outputs_tr[0].shape)
                # print(targets_tr.shape)
                loss_ops_tr = create_loss_ops(targets_tr, outputs_tr)
                loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

            gradients = tape.gradient(loss_op_tr, model.trainable_variables)
            # aggregate the gradients from the full batch.
            # this is not there for mirror strategy
            replica_ctx = tf.distribute.get_replica_context()
            gradients = replica_ctx.all_reduce("mean", gradients)

            optimizer.apply(gradients, model.trainable_variables)
            return loss_op_tr

        per_example_losses = strategy.experimental_run_v2(update_step, args=(dist_inputs, ))
        mean_loss = strategy.reduce("sum", per_example_losses, axis=None)
        # mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
        return mean_loss

    # compiled_train_step = tf.function(train_step, input_signature=(input_signature,) )

    def train_epoch(dataset):
        total_loss = 0.
        num_batches = 0
        for inputs in dataset:
            total_loss += train_step(inputs).numpy()
            num_batches += 1

        return total_loss/num_batches

    # with strategy.scope():
        # checkpoint
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