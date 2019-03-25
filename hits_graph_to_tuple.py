#!/usr/bin/env python
import os
import numpy as np
from functools import partial

from nx_graph import utils_data
from nx_graph import prepare

from datasets.graph import load_graph
import multiprocessing as mp

def process_event(evt_id, input_dir, n_sections, saver_fn, bidirection):
    for isec in range(n_sections):
        input_name = os.path.join(
            input_dir,
            'event{:09d}_g{:03d}.npz'.format(evt_id, isec))
        if os.path.exists(input_name) and not saver_fn(evt_id, isec, None):
            graph = utils_data.hitsgraph_to_networkx_graph(
                load_graph(input_name), bidirection=bidirection)

            saver_fn(evt_id, isec, graph)

if __name__ == "__main__":
    import numpy as np
    import glob
    import re

    from graph_nets import utils_np

    from nx_graph.utils_train import load_config

    import argparse

    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('config',  nargs='?', default='configs/nxgraph_default.yaml')
    add_arg('-b', '--bidirection', action='store_true')
    add_arg('-i', '--itask', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = config['data']['input_hitsgraph_dir']
    base_dir = os.path.join(input_dir,'event00000{}_g{:03d}.npz')

    # find number of files, section IDs in hitgraphs directory
    file_patten = base_dir.format(1000, 0).replace('1000', '*')
    all_files = glob.glob(file_patten)
    n_events = len(all_files)
    evt_ids = sorted([int(re.search('event00000([0-9]*)_g000.npz', os.path.basename(x)).group(1))
               for x in all_files])

    section_patten = base_dir.format(1000, 0).replace('_g000', '_g[0-9]*[0-9]*[0-9]')
    n_sections = len(glob.glob(section_patten))
    n_total = n_events*n_sections
    print("Total Events: {} with {} sections, total {} files ".format(
        n_events, n_sections, n_total))

    output = config['data']['output_nxgraph_dir']
    if not os.path.exists(output):
        os.makedirs(output)

    saver = prepare.get_networkx_saver(output)

    print("Input directory: {}".format(input_dir))
    print("Output directory: {}".format(output))

    import time
    start_time = time.time()

    log_name = os.path.join(output, config['data']['log_name'])

    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), # (event number), T (elapsed seconds)\n"
    with open(log_name, 'a') as f:
        f.write(out_str)

    out_str = ""
    # split all evt_ids to n_tasks and n_workers
    n_tasks = config['data']['n_tasks']
    n_workers = config['data']['n_workers']
    evt_ids_split = np.array_split(evt_ids, n_tasks)[args.itask]

    # invoke workers
    with mp.Pool(processes=n_workers) as pool:
        pp_fn = partial(process_event,
                        input_dir=input_dir,
                        n_sections=n_sections,
                        saver_fn=saver,
                        bidirection=args.bidirection)
        pool.map(pp_fn, evt_ids_split)

    elapsed_time = time.time() - start_time
    info = "# {:05d}, # {:05d}, T {:.1f}\n".format(ii, evt_id, elapsed_time)
    out_str += info
    print(info)

    with open(log_name, 'a') as f:
        f.write(out_str)
