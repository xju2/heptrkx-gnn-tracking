"""Handle read and write objects"""

import os
import glob
import re

import numpy as np
import pandas as pd
import yaml
import os

import networkx as nx
import sklearn.metrics

from graph_nets import utils_np

import matplotlib.pyplot as plt

def evtids_at_disk(evt_dir):
    all_files = glob.glob(os.path.join(evt_dir, '*hits*'))
    evtids = np.sort([int(
        re.search('event([0-9]*)', os.path.basename(x).split('-')[0]).group(1))
        for x in all_files])
    return evtids


def load_yaml(file_name):
    if not os.path.exists(file_name):
        raise NameError("{} not there".format(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def list_from_str(input_str):
    items = input_str.split(',')
    out = []
    for item in items:
        try:
            value = int(item)
            out.append(value)
        except ValueError:
            start, end = item.split('-')
            try:
                start, end = int(start), int(end)
                out += list(range(start, end+1))
            except ValueError:
                pass
    return out

import time

def read_log(file_name):
    time_format = '%d %b %Y %H:%M:%S'
    get2nd = lambda x: x.split()[1]

    time_info = []
    data_info = []
    itime = -1
    with open(file_name) as f:
        for line in f:
            if line[0] != '#':
                tt = time.strptime(line[:-1], time_format)
                time_info.append(tt)
                data_info.append([])
                itime += 1
            else:
                items = line.split(',')
                try:
                    iteration = int(get2nd(items[0]))
                except ValueError:
                    continue
                time_consumption = float(get2nd(items[1]))
                loss_train = float(get2nd(items[2]))
                loss_test  = float(get2nd(items[3]))
                precision  = float(get2nd(items[4]))
                recall     = float(get2nd(items[5]))
                data_info[itime].append([iteration, time_consumption, loss_train,
                                      loss_test, precision, recall])
    return data_info, time_info


def plot_log(info, name, axs=None):
    fontsize = 16
    minor_size = 14
    if type(info) is not 'numpy.ndarray':
        info = np.array(info)
    df = pd.DataFrame(info, columns=['iteration', 'time', 'loss_train', 'loss_test', 'precision', 'recall'])

    # make plots
    if axs is None:
        _, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axs = axs.flatten()

    y_labels = ['Time [s]', 'Training Loss', 'Precision', 'Recall']
    y_data   = ['time', 'loss_train', 'precision', 'recall']
    x_label = 'Iterations'
    x_data = 'iteration'
    for ib, values in enumerate(zip(y_data, y_labels)):
        ax = axs[ib]

        if 'loss_train' == values[0]:
            df.plot(x=x_data, y=values[0], ax=ax, label='Training')
            df.plot(x=x_data, y='loss_test', ax=ax, label='Testing')
            ax.set_ylabel("Losses", fontsize=fontsize)
            ax.legend(fontsize=fontsize)
        else:
            df.plot(x=x_data, y=values[0], ax=ax)
            ax.set_ylabel(values[1], fontsize=fontsize)

        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    return axs


def is_df_there(file_name):
    res = False
    if os.path.exists(file_name):
        try:
            _ = pd.read_hdf(file_name, mode='r')
        except ValueError:
            pass
        else:
            res = True
    return res

def is_diff_networkx(g1, g2):
    m1 = nx.to_numpy_matrix(g1)
    m2 = nx.to_numpy_matrix(g2)
    return m1 == m2


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


def np_to_nx(array, hits):
    G = nx.Graph()

    node_features = ['r', 'phi', 'z']

    used_hits = array['I']
    df = pd.DataFrame(used_hits, columns=['hit_id'])
    df = df.merge(hits[['hit_id']+node_features], on='hit_id')
    node_info = [
        (i, dict(pos=np.array(row[1:]), hit_id=row[0])) for i,row in df.iterrows()
    ]
    G.add_nodes_from(node_info)

    receivers = array['receivers']
    senders = array['senders']
    score = array['score']
    truth = array['truth']
    edge_info = [
        (i, j, dict(weight=k, solution=l)) for i,j,k,l in zip(senders, receivers, score, truth)
    ]
    G.add_edges_from(edge_info)
    return G


def count_total_params(reader, count_exclude_pattern=""):
  """Count total number of variables."""
  var_to_shape_map = reader.get_variable_to_shape_map()

  # Filter out tensors that we don't want to count
  if count_exclude_pattern:
    regex_pattern = re.compile(count_exclude_pattern)
    new_var_to_shape_map = {}
    exclude_num_tensors = 0
    exclude_num_params = 0
    for v in var_to_shape_map:
      if regex_pattern.search(v):
        exclude_num_tensors += 1
        exclude_num_params += np.prod(var_to_shape_map[v])
      else:
        new_var_to_shape_map[v] = var_to_shape_map[v]
    var_to_shape_map = new_var_to_shape_map
    print("# Excluding %d tensors (%d params) that match %s when counting." % (
        exclude_num_tensors, exclude_num_params, count_exclude_pattern))

  var_sizes = [np.prod(var_to_shape_map[v]) for v in var_to_shape_map]
  return np.sum(var_sizes, dtype=int)