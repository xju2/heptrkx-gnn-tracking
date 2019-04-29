#!/usr/bin/env python3

def load_seg(file_name):
    with pd.HDFStore(file_name) as store:
        return store.get('data')

if __name__ == "__main__":
    evtid = 6600
    selected_pairs_dir = '/global/cscratch1/sd/xju/heptrkx/output_graphs/selected_pairs/evt{}'.format(evtid)

    import os
    import glob
    import pandas as pd


    from make_true_pairs_for_training_segments_mpi import layer_pairs

    all_segments = []

    selected_pairs_files = [os.path.join(selected_pairs_dir, 'selected_pair{:03d}.h5'.format(x)) for x in range(90)]

    out = ''
    for ii, file_name in enumerate(selected_pairs_files):
        segment = load_seg(file_name)

        df_selected = segment[segment.selected]
        n_selected = df_selected.shape[0]
        n_true = segment[segment.true].shape[0]
        n_true_selected = segment[ (segment.true) & (segment.selected)].shape[0]
        start, end = layer_pairs[ii]
        out += "{:2} {:2} {:2} {:10} {:10} {:10} {:10}\n".format(ii, start, end, segment.shape[0], n_true, n_selected, n_true_selected)
        #out = "{:2} {:2} {:2} {:10} {:10} {:10} {:10}\n".format(ii, start, end, segment.shape[0], n_true, n_selected, n_true_selected)
        #print(out)

        all_segments.append(segment.shape[0])

    sum_segments = sum(all_segments)
    print(sum_segments)

    with open("log.segments_660", 'w') as f:
        f.write(out)

