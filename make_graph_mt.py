#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    import pandas as pd
    import subprocess

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('data_dir', type=str, help='event directory')
    add_arg('blacklist_dir', type=str, help='blacklist directory')
    add_arg('evtid', type=int, help='event id')
    add_arg('model_weight_dir', type=str, help='directory containing model weights')
    add_arg('threshold_file', type=str, help='text file containing threshold')
    add_arg('output_dir', type=str, help='save created graph')

    args = parser.parse_args()

    data_dir = args.data_dir
    black_list_dir = args.blacklist_dir
    evtid = args.evtid
    output_dir = args.output_dir
    model_weight_dir = args.model_weight_dir
    threshold_file = args.threshold_file

    df_threshold = pd.read_csv(threshold_file, sep=' ', header=None,
                               names=["idx", "in", "out", "cut", "eff", "purity", "n_true", "n_fake"])
    print(df_threshold.shape)


    os.makedirs(output_dir, exist_ok=True)

    srun_cmd_base = ['srun', '-n', '1', '-c', '32', 'python']

    raw_pairs_dir = os.path.join(output_dir, 'raw_pairs')
    cmd_raw_pairs = srun_cmd_base + [
        'make_pairs_for_training_segments.py', data_dir, black_list_dir, str(evtid), raw_pairs_dir]

    print(" ".join(cmd_raw_pairs))
    #ck_code = subprocess.call(cmd_raw_pairs)
    #print('results', ck_code)
    #exit()

    from make_pairs_for_training_segments_mpi import layer_pairs

    selected_pairs_dir = os.path.join(output_dir, 'selected_pairs', 'evt{}'.format(evtid))
    for ii, layer_pair in enumerate(layer_pairs):
        file_name = os.path.join(raw_pairs_dir, 'evt{}'.format(evtid), 'pair{:03d}.h5'.format(ii))
        cmd_sel_pairs =['python', 'select_pairs.py', file_name,
                        os.path.join(model_weight_dir, 'modelpair{:03d}.ckpt'.format(ii)),
                        str(df_threshold[df_threshold.idx == ii]['cut'].values[0]),
                        selected_pairs_dir ]
        print(" ".join(cmd_sel_pairs))
        ck_code = subprocess.call(cmd_sel_pairs)
        print("pair selection:", ck_code)
