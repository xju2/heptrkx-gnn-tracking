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
    add_arg('config', nargs='?', default='configs/train_pairs.yaml')
    add_arg('--resume-train',  action='store_true')
    add_arg('--in-eval', action='store_true')
    add_arg('--file-name', type=str, default=None)
    add_arg('--truth-file', type=str, default=None)

    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.load(f)


    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    output_dir = config['output_dir']
    file_name = config['data']['file_name']
    if args.file_name is not None:
        file_name = args.file_name

    ## save checkpoints
    pairs_base_name = os.path.basename(file_name)
    checkpoint_path = os.path.join(output_dir, "model{}".format(pairs_base_name.replace('h5', 'ckpt')))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # information output
    pair_idx = int(pairs_base_name.replace('.h5', '')[4:])

    from make_pairs_for_training_segments import layer_pairs
    layer_info = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
    pair_info = layer_info[pair_idx]

    outname = os.path.join(output_dir, 'info{:03d}-{}-{}.txt'.format(pair_idx, *pair_info))
    out_predictions = os.path.join(output_dir, 'test_prediction.h5')

    if os.path.exists(checkpoint_path+".index") and not args.resume_train and os.path.exists(outname) and os.path.exists(out_predictions):
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
    with pd.HDFStore(file_name) as store:
        df_input = store['data'].astype(np.float64)


    true_file = config['data']['truth_file']
    if args.truth_file is not None:
        true_file = args.truth_file

    if true_file is not 'None':
        from sklearn.utils import shuffle
        with pd.HDFStore(true_file) as store:
            df_true = store['data'].astype(np.float64)
            df_input = pd.concat([df_input, df_true], ignore_index=True)
            df_input = shuffle(df_input, random_state=10)

    df_input = keep_finite(df_input)
    features = config['data']['features']


    all_inputs  = df_input[features].values
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
    all_inputs_normed = scaler.fit_transform(all_inputs)

    inputs = all_inputs_normed[:n_training, :]
    targets = all_targets[:n_training, :]

    x_val = all_inputs_normed[n_training:n_training+n_validating, :]
    y_val = all_targets[n_training:n_training+n_validating, :]

    x_test = all_inputs_normed[n_training+n_validating:, :]
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

    test_inputs = df_input[n_training+n_validating:]
    test_inputs = test_inputs.assign(prediction=prediction)


    from nx_graph.utils_plot import plot_metrics

    plot_metrics(prediction, y_test,
                 outname=os.path.join(output_dir, 'roc_{}-{}.png'.format(*pair_info)))

    # find a threshold
    from sklearn.metrics import precision_recall_curve
    y_true = y_test > 0.5
    purity, efficiency, thresholds = precision_recall_curve(y_true, prediction)
    #print(len(purity), len(efficiency), len(thresholds))

    eff_cut = config['prediction']['eff_cut']
    from bisect import bisect
    ti = bisect(list(reversed(efficiency.tolist())), eff_cut)
    ti = len(efficiency) - ti
    thres = thresholds[ti]
    out = "{} {} {} {th:.4f} {tp:.4f} {fp:.4f} {true} {fake}\n".format(
        pair_idx, *pair_info, th=thres, tp=efficiency[ti], fp=purity[ti],
        true=n_true, fake=n_fake)

    with open(outname, 'a') as f:
        f.write(out)


    # save prediction and test result
    with pd.HDFStore(out_predictions) as store:
        store['data'] = test_inputs
