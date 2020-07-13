
#!/usr/bin/env python
"""
Training GNN in TPU/GPU/CPU with fixed graph size
Its input is TFRecord
"""

import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("Version:{}".format(tf.__version__))
# tf.debugging.set_log_device_placement(True)

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
# from heptrkx.nx_graph.distribute_model import SegmentClassifier
# from heptrkx.nx_graph.model import SegmentClassifier
from heptrkx.nx_graph import get_model
from heptrkx.utils import load_yaml

prog_name = os.path.basename(sys.argv[0])

def train_and_evaluate(args):
    use_tpu = args.tpu is not None

    device = 'CPU'
    global_batch_size = args.train_batch_size if not use_tpu else args.tpu_cores
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    n_gpus = len(physical_gpus)

    if use_tpu:
        if args.tpu == 'colab':
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        else:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu, zone=args.zone)
        workers = resolver.cluster_spec().as_dict()['worker']
        n_tpus = len(workers)
        logging.info('Running on {} TPUs '.format(n_tpus))
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = snt.distribute.TpuReplicator(resolver)
        device = 'TPU'
    elif n_gpus > 1:
        logging.info("Useing SNT Replicator with {} GPUs".format(n_gpus))
        strategy = snt.distribute.Replicator(['/device:GPU:{}'.format(i) for i in range(n_gpus)],\
            tf.distribute.ReductionToOneDevice("CPU:0"))
        device = "{}GPUs".format(n_gpus)
        for dd in physical_gpus:
            tf.config.experimental.set_memory_growth(dd, True)
    elif n_gpus > 0:
        strategy = tf.distribute.OneDeviceStrategy("/device:GPU:0")
        device = "1GPU"
    else:
        strategy = tf.distribute.OneDeviceStrategy("/device:CPU:0")

    if n_gpus > 0:
        assert n_gpus == global_batch_size, "batch size {} does not equall to GPUs {}".format(global_batch_size, n_gpus)
    else:
        pass
        # assert global_batch_size == 1, "batch size {} does not equall to 1".format(global_batch_size)

    output_dir = args.job_dir
    if not use_tpu:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    num_processing_steps_tr = args.num_iters     ## level of message-passing
    n_epochs = args.num_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, global_batch_size))
    logging.info("{} processing steps in the model".format(num_processing_steps_tr))
    # prepare graphs
    logging.info("{} Eta bins and {} Phi bins".format(args.num_eta_bins, args.num_phi_bins))
    max_nodes, max_edges = graph.get_max_graph_size(args.num_eta_bins, args.num_phi_bins)

    train_files = tf.io.gfile.glob(args.train_files)
    eval_files = tf.io.gfile.glob(args.eval_files)

    # logging.info("Input file names: ", file_names)
    logging.info("{} training files and {} evaluation files".format(len(train_files), len(eval_files)))
    logging.info("{} training files".format(n_train))
    raw_dataset = tf.data.TFRecordDataset(file_names)
    training_dataset = raw_dataset.map(graph.parse_tfrec_function)

    AUTO = tf.data.experimental.AUTOTUNE
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    training_dataset = training_dataset.with_options(options)
    training_dataset = training_dataset.batch(global_batch_size, drop_remainder=True).prefetch(AUTO)

    learning_rate = args.learning_rate
    with strategy.scope():
        optimizer = snt.optimizers.Adam(learning_rate)
        # model = SegmentClassifier()
        model = get_model(args.model_name)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=5)
    logging.info("Loading latest checkpoint from:", output_dir)
    status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    # training loss
    real_weight = args.real_edge_weight
    fake_weight = args.fake_edge_weight

    def log_loss(label, prediction, eps=1e-7, weights=1.):
        # tf.compat.v1.losses.log_loss, not supported by TPU
        # copy the TF source code here
        loss = tf.negative(tf.add(
            tf.multiply(label, tf.math.log(prediction+eps)),
            tf.multiply((1 - label), tf.math.log(1 - prediction+eps))))
        loss = tf.multiply(loss, weights)
        present = tf.where(
            tf.math.equal(weights, 0.0), tf.zeros_like(weights), tf.ones_like(weights))
        loss = tf.math.divide_no_nan(tf.math.reduce_sum(loss), tf.math.reduce_sum(present))
        return loss

    def create_loss_ops(target_op, output_ops):
        weights = target_op.edges * real_weight + (1 - target_op.edges) * fake_weight
        row_index = tf.range(tf.constant(max_edges))
        n_valid_edges = target_op.n_edge[0]
        
        # # NOTE: this implementation is very low
        # # cond = (row_index < n_valid_edges)
        # # zeros = tf.zeros_like(weights, dtype=weights.dtype)
        # # weights = tf.where(cond, weights, zeros)

        mask = tf.cast(row_index < n_valid_edges, tf.float32)
        mask = tf.expand_dims(mask, axis=1)
        weights = weights * mask

        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.edges, output_op.edges, weights=weights)
                for output_op in output_ops
        ]

        return tf.stack(loss_ops)

    @tf.function(autograph=False)
    def train_step(inputs_tr, targets_tr):
        logging.info("Tracing train_step")
        logging.info(inputs_tr)

        def update_step(inputs_tr, targets_tr):
            logging.info("Tracing update_step")
            logging.info("before contatenate:", inputs_tr.n_node.shape)
            inputs_tr = graph.concat_batch_dim(inputs_tr)
            targets_tr = graph.concat_batch_dim(targets_tr)
            logging.info("after concatenate:", inputs_tr.n_node.shape)

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

        per_example_losses = strategy.run(update_step, args=(inputs_tr,targets_tr))
        mean_loss = strategy.reduce("sum", per_example_losses, axis=None)
        return mean_loss

    def train_epoch(dataset):
        total_loss = 0.
        num_batches = 0
        for inputs in dataset:
            input_tr, target_tr = inputs
            total_loss += train_step(input_tr, target_tr)
            num_batches += 1
        logging.info("total batches:", num_batches)
        return total_loss/num_batches

    this_time =  time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str  = "Start training " + time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "Epoch, Time [mins], Loss\n"
    log_name = os.path.join(output_dir, "training_log.txt")
    with open(log_name, 'a') as f:
        f.write(out_str)
    now = time.time()
    # writer = tf.summary.create_file_writer(os.path.join(output_dir, this_time))

    for epoch in range(n_epochs):
        logging.info("start epoch {} on {}".format(epoch, device))
        training_dataset = training_dataset.shuffle(global_batch_size*2, reshuffle_each_iteration=True)
        dist_training_dataset = strategy.experimental_distribute_dataset(training_dataset)
        loss = train_epoch(dist_training_dataset)
        # loss = train_epoch(training_dataset)
        this_epoch = time.time()
        
        logging.info("Training {} epoch, {:.2f} mins, Loss := {:.4f}".format(
            epoch, (this_epoch-now)/60., loss/global_batch_size))
        out_str = "{}, {:.2f}, {:.4f}\n".format(epoch, (this_epoch-now)/60., loss/global_batch_size)
        with open(log_name, 'a') as f:
            f.write(out_str)
        # with writer.as_default():
        #     tf.sumary.scalar("")

        now = this_epoch
        ckpt_manager.save()

    out_log = "End @ " + time.strftime('%d %b %Y %H:%M:%S', time.localtime()) + "\n"
    with open(log_name, 'a') as f:
        f.write(out_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("--train-files", help='path to training data', required=True)
    add_arg("--eval-files", help='path to evaluation data', required=True)
    add_arg("--job-dir", help='location to write checkpoints and export models', required=True)
    add_arg("--train-batch-size", help='batch size for training', default=2, type=int)
    add_arg("--eval-batch-size", help='batch size for evaluation', default=2, type=int)
    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg("--learning-rate", help='learing rate', default=0.0005, type=float)
    add_arg("--num-epochs", help='number of epochs', default=1, type=int)
    add_arg("--model-name", help='model name', default='vary2')

    add_arg("--num-eta-bins", default=1, help='number of eat bins', type=int)
    add_arg("--num-phi-bins", default=1, help='number of phi bins', type=int)

    add_arg("--real-edge-weight", help='weights for real edges', default=2., type=float)
    add_arg("--fake-edge-weight", help='weights for fake edges', default=1., type=float)

    add_arg('--tpu', help='use tpu', default=None)
    add_arg("--tpu-cores", help='number of cores in TPU', default=8, type=int)
    add_arg('--zone', help='gcloud zone for tpu', default='us-central1-b')
    add_arg("-v", "--verbose", )
    args, _ = parser.parse_known_args()

    # Set python level verbosity
    logging.set_verbosity(args.verbosity)
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train_and_evaluate(args)