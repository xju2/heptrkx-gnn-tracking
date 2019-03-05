#!/usr/bin/env python

import numpy as np
import sklearn.metrics

import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import utils_np

from nx_graph.utils_train import create_feed_dict
from nx_graph.utils_train import create_loss_ops
from nx_graph.utils_train import make_all_runnable_in_session
from nx_graph.utils_train import compute_matrics
from nx_graph.utils_train import load_config


if __name__ == "__main__":
    import sys
    import argparse
    import glob
    import re

    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('config',  nargs='?', default='configs/nxgraph_default.yaml')

    args = parser.parse_args()

    config = load_config(args.config)

    import time
    import os

    from nx_graph.prepare import inputs_generator
    # default 2/3 for training and 1/3 for testing
    generate_input_target = inputs_generator(config['data']['output_nxgraph_dir'], n_train_fraction=0.8)

    config_tr = config['train']
    # How much time between logging and printing the current results.
    # save checkpoint very 10 mins
    log_every_seconds       = config_tr['time_lapse']
    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_training_iterations = config_tr['iterations']
    iter_per_job            = 2500 if 'iter_per_job' not in config_tr else config_tr['iter_per_job']
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    print("Maximum iterations per job: {}".format(iter_per_job))

    # add ops to save and restore all the variables
    output_dir = os.path.join(config['output_dir'], prod_name)
    print("trained model will be save at:", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ## start to build tensorflow sessions
    tf.reset_default_graph()

    #from nx_graph.model_mlp import SegmentClassifier
    #model = SegmentClassifier()
    from nx_graph import get_model
    model = get_model(config['model']['name'])

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)

    # Training loss.
    loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

    # Optimizer
    learning_rate = config_tr['learning_rate']
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    # copyed from deepmind's example, not sure needed...
    input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

    sess = tf.Session()

    files = glob.glob(output_dir+"/*.ckpt.meta")
    last_iteration = 0 if len(files) < 1 else max([
        int(re.search('checkpoint_([0-9]*).ckpt.meta', os.path.basename(x)).group(1))
        for x in files
    ])
    print("last iteration:", last_iteration)

    saver = tf.train.Saver()
    ckpt_name = 'checkpoint_{:05d}.ckpt'
    if last_iteration > 0:
        print("loading checkpoint:", os.path.join(output_dir, ckpt_name.format(last_iteration)))
        saver.restore(sess, os.path.join(output_dir, ckpt_name.format(last_iteration)))
    else:
        init_ops = tf.global_variables_initializer()
        # saver must be created before init_ops is run!
        sess.run(init_ops)

    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []


    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), T (elapsed seconds), Ltr (training loss), Precision, Recall\n"
    log_name = os.path.join(output_dir, config_tr['log_name'])
    with open(log_name, 'a') as f:
        f.write(out_str)

    start_time = time.time()
    last_log_time = start_time

    ## loop over iterations, each iteration generating a batch of data for training
    iruns = 0
    all_run_time = start_time
    all_data_taking_time = start_time

    print("# (iteration number), TD (get graph), TR (TF run)")
    for iteration in range(last_iteration, num_training_iterations):
        if iruns > iter_per_job:
            print("runs larger than {} iterations per job, stop".format(iter_per_job))
            break
        else: iruns += 1
        last_iteration = iteration
        data_start_time = time.time()
        feed_dict = create_feed_dict(generate_input_target, batch_size, input_ph, target_ph)
        all_data_taking_time += time.time() - data_start_time

        # timing the run time only
        run_start_time = time.time()
        train_values = sess.run({
            "step": step_op,
            "target": target_ph,
            "loss": loss_op_tr,
            "outputs": output_ops_tr
        }, feed_dict=feed_dict)
        run_time = time.time() - run_start_time
        all_run_time += run_time

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        if elapsed_since_last_log > log_every_seconds:
            # save a checkpoint
            last_log_time = the_time
            feed_dict = create_feed_dict(generate_input_target,
                                         batch_size, input_ph, target_ph, is_trained=False)
            test_values = sess.run({
                "target": target_ph,
                "loss": loss_op_tr,
                "outputs": output_ops_tr
            }, feed_dict=feed_dict)
            correct_tr, solved_tr = compute_matrics(
                test_values["target"], test_values["outputs"][-1])
            elapsed = time.time() - start_time
            losses_tr.append(train_values["loss"])
            corrects_tr.append(correct_tr)
            solveds_tr.append(solved_tr)
            logged_iterations.append(iteration)
            out_str = "# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Precision {:.4f}, Recall {:.4f}\n".format(
                iteration, elapsed, train_values["loss"], test_values["loss"],
                correct_tr, solved_tr)

            run_cost_time = all_run_time - start_time
            data_cost_time = all_data_taking_time - start_time
            print("# {:05d}, TD {:.1f}, TR {:.1f}".format(iteration, data_cost_time, run_cost_time))
            with open(log_name, 'a') as f:
                f.write(out_str)

            save_path = saver.save(
                sess,
                os.path.join(output_dir, ckpt_name.format(iteration)))
    sess.close()
