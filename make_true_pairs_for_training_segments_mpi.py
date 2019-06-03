#!/usr/bin/env python3

layer_pairs = [
    (7, 8), (8, 9), (9, 10), (10, 24), (24, 25), (25, 26), (26, 27), (27, 40), (40, 41),
    (7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0),
    (8, 6), (9, 6),
    (7, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    (8, 11), (9, 11),
    (24, 23), (23, 22), (22, 21), (21, 19), (19, 18),
    (24, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
    (25, 23), (26, 23), (25, 28), (26, 28),
    (27, 39), (40, 39), (27, 42), (40, 42),
    (39, 38), (38, 37), (37, 36), (36, 35), (35, 34),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
    (19, 34), (20, 35), (21, 36), (22, 37), (23, 38),
    (28, 43), (29, 44), (30, 45), (31, 46), (32, 47),
    (0, 18), (0, 19), (1, 20), (1, 21), (2, 21), (2, 22), (3, 22), (4, 23),
    (17, 33), (17, 32), (17, 31), (16, 31), (16, 30), (15, 30), (15, 29), (14, 29), (14, 28), (13, 29), (13, 28),
    (11, 24), (12, 24), (6, 24), (5, 24), (4, 24)
]

# bad_layer_pairs = [
#     (8, 6), (7, 6)
# ]
# layer_pairs = bad_layer_pairs

layer_pairs_dict = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
pairs_layer_dict = dict([(layer_pair, ii) for ii, layer_pair in enumerate(layer_pairs)])


def process(input_info, selected_hits_angle, output_dir):
    layer_pair, ii = input_info
    segments = list(utils_mldata.create_segments(selected_hits_angle, [layer_pair], only_true=True))
    os.makedirs(output_dir, exist_ok=True)

    with pd.HDFStore(os.path.join(output_dir, 'pair{:03d}.h5'.format(ii))) as store:
            store['data'] = segments[0]


if __name__ == "__main__":
    import os
    import glob
    import re
    import yaml

    import argparse

    parser = argparse.ArgumentParser(description='produce true pairs using MPI')
    add_arg = parser.add_argument
    add_arg('config', type=str, help='data configuration, configs/data.yaml')

    args = parser.parse_args()

    assert(os.path.exists(args.config))
    with open(args.config) as f:
        config = yaml.load(f)

    data_dir = os.path.expandvars(config['track_ml']['dir'])
    black_list_dir = os.path.expandvars(config['track_ml']['blacklist_dir'])
    det_dir  = os.path.expandvars(config['track_ml']['detector'])
    base_output_dir = os.path.expandvars(config['true_hits']['dir'])

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print("World size:", size, ", rank:", rank)
        use_mpi = True
    except ImportError:
        rank = 0
        size = 1
        use_mpi = False

    from preprocess import utils_mldata
    import numpy as np
    import pandas as pd
    from nx_graph import utils_data

    if rank == 0:
        all_files = glob.glob(os.path.join(data_dir, '*-hits.csv'))
        evt_ids = np.sort([int(re.search('event00000([0-9]*)-hits.csv',
                                         os.path.basename(x)).group(1))
                           for x in all_files])
        n_events = len(evt_ids)
        print("Total Events:", n_events)

        # remove existing ones
        all_existing_files = glob.glob(os.path.join(base_output_dir, '*'))
        existing_evt_ids = set([int(os.path.basename(x)[3:]) for x in all_existing_files])
        set_evt_ids = set(evt_ids.tolist())
        evt_ids = np.array(list(set_evt_ids.difference(existing_evt_ids)))
        print("Left Events:", len(evt_ids))

        ## check existing evt-ids
        evt_ids = [x.tolist() for x in np.array_split(evt_ids, size)]

    else:
        evt_ids = None

    if use_mpi:
        comm.Barrier()
        evt_ids = comm.scatter(evt_ids, root=0)
    else:
        evt_ids = evt_ids[0]

    from nx_graph import transformation
    module_getter = utils_mldata.module_info(det_dir)

    from functools import partial
    import multiprocessing as mp

    try:
        n_workers = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except (ValueError, TypeError):
        n_workers = 1

    print(rank, "# workers:", n_workers)
    print(rank, "# evts:", len(evt_ids))

    for evtid in evt_ids:
        output_dir = os.path.join(base_output_dir, 'evt{}'.format(evtid))
        if os.path.exists(output_dir):
            continue
        hits, particles, truth, cells = utils_mldata.read(data_dir, black_list_dir, evtid)

        reco_pids = utils_mldata.reconstructable_pids(particles, truth)

        # noise included!
        hh = utils_data.merge_truth_info_to_hits(hits, truth, particles)
        unique_pids = np.unique(hh['particle_id'])
        print("Event", evtid, "Number of particles:", unique_pids.shape, reco_pids.shape)

        n_pids = len(unique_pids)
        selected_pids = np.random.choice(unique_pids, size=n_pids)
        selected_hits = hh[hh.particle_id.isin(selected_pids)].assign(evtid=evtid)

        local_angles = utils_mldata.cell_angles(selected_hits, module_getter, cells)
        selected_hits_angle = selected_hits.merge(local_angles, on='hit_id', how='left')

        pp_layers_info = [(x, ii) for ii,x in enumerate(layer_pairs)]

        with mp.Pool(processes=n_workers) as pool:
            pp_func=partial(process, selected_hits_angle=selected_hits_angle, output_dir=output_dir)
            pool.map(pp_func, pp_layers_info)
