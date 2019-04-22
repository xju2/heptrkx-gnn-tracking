#!/usr/bin/env python3

import numpy as np

def keep_finite(df):
    bad_list = []
    for column in df.columns:
        if not np.all(np.isfinite(df[column])):
            ss = df[column]
            bad_list += ss.loc[~np.isfinite(ss)].index.values.tolist()

    bad_list = list(set(bad_list))
    return df.drop(bad_list)

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('file_name',  nargs='?', default='~/atlas/heptrkx/trackml_inputs/doublet_candidates_for_training/all_pairs/evt6600/pair000.h5',
            help='file name for pair candidates')
    add_arg('--batch-size',  type=int, default=64)
    add_arg('--epochs',  type=int, default=1)
    add_arg('--resume-train',  action='store_true')
    add_arg('--true-file', type=str, default=None)
    add_arg('--in-eval', action='store_true')
    add_arg('--eff-cut', type=float, default=0.98, help='threshold that renders such efficiency')
    #add_arg('--true-file', type=str, default='/global/cscratch1/sd/xju/heptrkx/pairs/merged_true_pairs/training/pair000.h5')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs


    output_dir = os.path.join('trained_results', 'doublets')
    ## save checkpoints
    pairs_base_name = os.path.basename(args.file_name)
    checkpoint_path = os.path.join(output_dir, "model{}".format(pairs_base_name.replace('h5', 'ckpt')))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # information output
    pair_idx = int(pairs_base_name.replace('.h5', '')[4:])

    from make_pairs_for_training_segments import layer_pairs
    layer_info = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
    pair_info = layer_info[pair_idx]

    outname = os.path.join(output_dir, 'info{:03d}-{}-{}.txt'.format(pair_idx, *pair_info))

    if os.path.exists(checkpoint_path+".index") and not args.resume_train and os.path.exists(outname):
        print("model is trained and evaluated")
        exit()

    import tensorflow as tf
    from tensorflow import keras

    # same model for everyone
    model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)])

    if os.path.exists(checkpoint_path+".index") and (args.resume_train or args.in_eval):
        print("Resume previous training")
        model.load_weights(checkpoint_path)


    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    import pandas as pd
    import numpy as np
    #df_input = pd.read_csv(args.file_name)
    with pd.HDFStore(args.file_name) as store:
        df_input = store['data'].astype(np.float64)


    if args.true_file:
        from sklearn.utils import shuffle
        with pd.HDFStore(args.true_file) as store:
            df_true = store['data'].astype(np.float64)
            df_input = pd.concat([df_input, df_true], ignore_index=True)
            df_input = shuffle(df_input)
            #df_input = df_input.sample(frac=1).reset_index(drop=True)

    #df_input['true'] = df_input['true'].astype(np.int32)
    df_input = keep_finite(df_input)


    all_inputs  = df_input[['dphi', 'dz', 'dr', 'phi_slope', 'z0', 'deta', 'deta1', 'dphi1']].values
    all_targets = df_input[['true']].values
    n_total = all_inputs.shape[0]
    n_true = np.sum(all_targets)
    n_fake = n_total - n_true
    print("All Entries:", n_total)
    print("True:", n_true)
    print("Fake:", n_fake)

    n_training = int(n_total*0.8)
    n_validating = int(n_total*0.1)

    # transform all inputs
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_inputs = scaler.fit_transform(all_inputs)

    inputs = all_inputs[:n_training, :]
    targets = all_targets[:n_training, :]

    x_val = all_inputs [n_training:n_training+n_validating, :]
    y_val = all_targets[n_training:n_training+n_validating, :]

    x_test = all_inputs[n_training+n_validating:, :]
    y_test = all_targets[n_training+n_validating:, :]


    if not args.in_eval:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = model.fit(inputs, targets,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val),
                            callbacks = [cp_callback, early_stop],
                            class_weight={0: 1, 1: n_fake/n_true},
                            verbose=1)

    prediction = model.predict(x_test,
                               batch_size=batch_size)


    from nx_graph.utils_plot import plot_metrics

    plot_metrics(prediction, y_test,
                 outname=os.path.join(output_dir, 'roc_{}-{}.png'.format(*pair_info)))

    # find a threshold
    from sklearn.metrics import precision_recall_curve
    y_true = y_test > 0.5
    purity, efficiency, thresholds = precision_recall_curve(y_true, prediction)
    #print(len(purity), len(efficiency), len(thresholds))

    from bisect import bisect
    ti = bisect(list(reversed(efficiency.tolist())), args.eff_cut)
    ti = len(efficiency) - ti
    thres = thresholds[ti]
    out = "{} {} {} {th:.4f} {tp:.4f} {fp:.4f} {true} {fake}".format(
        pair_idx, *pair_info, th=thres, tp=efficiency[ti], fp=purity[ti],
        true=n_true, fake=n_fake)

    with open(outname, 'a') as f:
        f.write(out)
