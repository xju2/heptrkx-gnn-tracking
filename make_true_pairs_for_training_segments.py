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
layer_pairs_dict = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
pairs_layer_dict = dict([(layer_pair, ii) for ii, layer_pair in enumerate(layer_pairs)])


def process(input_info, selected_hits_angle):
    layer_pair, ii = input_info
    segments = list(utils_mldata.create_segments(selected_hits_angle, [layer_pair], only_true=True))

    #os.makedirs(output_pairs_dir, exist_ok=True)
    #with pd.HDFStore(os.path.join(output_pairs_dir, 'pair{:03d}.h5'.format(ii))) as store:
    #        store['data'] = segments[0]
    return segments[0]


def merge_and_save(all_segments, output_pairs_dir):
    merged_segments = pd.concat([x[x.true] for x in all_segments], ignore_index=True)

    os.makedirs(output_pairs_dir, exist_ok=True)
    with pd.HDFStore(os.path.join(output_pairs_dir, 'pair{:03d}.h5'.format(ii))) as store:
            store['data'] = merged_segments


if __name__ == "__main__":
    data_dir = '/global/homes/x/xju/atlas/heptrkx/trackml_inputs/train_all'
    black_list_dir = '/global/homes/x/xju/atlas/heptrkx/trackml_inputs/blacklist'

    import os
    from preprocess import utils_mldata
    import glob
    import re

    import numpy as np
    import pandas as pd
    from nx_graph import utils_data

    all_files = glob.glob(os.path.join(data_dir, '*-hits.csv'))
    evt_ids = np.sort([int(re.search('event00000([0-9]*)-hits.csv',
                             os.path.basename(x)).group(1))
               for x in all_files])
    n_events = len(evt_ids)
    print("Total Events:", n_events)
    print(evt_ids[0])


    from nx_graph import transformation
    det_dir  = '/global/homes/x/xju/atlas/heptrkx/trackml_inputs/detectors.csv'
    module_getter = utils_mldata.module_info(det_dir)

    from functools import partial
    import multiprocessing as mp

    all_segments = []
    for evtid in evt_ids:
        hits, particles, truth, cells = utils_mldata.read(data_dir, black_list_dir, evtid)

        reco_pids = utils_mldata.reconstructable_pids(particles, truth)

        # noise included!
        hh = utils_data.merge_truth_info_to_hits(hits, truth, particles)
        unique_pids = np.unique(hh['particle_id'])
        print("Event", evtid, "Number of particles:", unique_pids.shape, reco_pids.shape)

        n_pids = len(unique_pids)
        selected_pids = np.random.choice(unique_pids, size=n_pids)
        selected_hits = hh[hh.particle_id.isin(selected_pids)].assign(evtid=evtid)
        # all_layers = np.unique(selected_hits.layer)
        # print("Total Number of Layers:", len(all_layers))
        # print("Total Layer Pairs", len(layer_pairs))



        local_angles = utils_mldata.cell_angles(selected_hits, module_getter, cells)
        selected_hits_angle = selected_hits.merge(local_angles, on='hit_id', how='left')

        pp_layers_info = [(x, ii) for ii,x in enumerate(layer_pairs)]
        n_workers = len(layer_pairs)
        print("Workers:", n_workers)

        with mp.Pool(processes=n_workers) as pool:
            pp_func=partial(process, selected_hits_angle=selected_hits_angle)
            segments = pool.map(pp_func, pp_layers_info)

        all_segments.append(segments)

    pp_segments = []
    for ilayer in range(len(layer_pairs)):
        pp_segments.append([x[ilayer] for x in all_segments])

    ## merge segments of each layer pair in all events
    with mp.Pool(processes=n_events) as pool:
        merge_fnc=partial(merge_and_save, output_pairs_dir=os.path.join('input_pairs', 'all_evts'))
        pool.map(merge_fnc, pp_segments)
