#!/usr/bin/env python3


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('file_name',  nargs='?', default='input_pairs/pair000.h5', help='file name for pair candidates')
    add_arg('--batch-size',  type=int, default=64)
    add_arg('--epochs',  type=int, default=1)
    add_arg('--resume-train',  action='store_true')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs


    import tensorflow as tf
    from tensorflow import keras

    ## save checkpoints
    pairs_base_name = os.path.basename(args.file_name)
    checkpoint_path = "trained_results/doublets/model{}".format(pairs_base_name.replace('h5', 'ckpt'))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # same model for everyone
    model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)])

    if args.resume_train and os.path.exists(checkpoint_path+".index"):
        print("Resume previous training")
        model.load_weights(checkpoint_path)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    import pandas as pd
    import numpy as np
    #df_input = pd.read_csv(args.file_name)
    with pd.HDFStore(args.file_name) as store:
        df_input = store['data']

    all_inputs = df_input[['dphi', 'dz', 'dr', 'phi_slope', 'z0', 'deta', 'deta1', 'dphi1']].values
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


    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001)

    history = model.fit(inputs, targets,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks = [cp_callback, early_stop],
                        class_weight={0: 1, 1: n_fake/n_true},
                        verbose=1)

    prediction = model.predict(x_test,
                               batch_size=batch_size)


    from nx_graph.utils_plot import plot_metrics
    from make_pairs_for_training_segments import layer_pairs
    layer_info = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
    pair_idx = int(pairs_base_name.replace('.h5', '')[4:])
    pair_info = layer_info[pair_idx]

    output_dir = os.path.join('trained_results', 'doublets')
    plot_metrics(prediction, y_test,
                 outname=os.path.join(output_dir, 'roc_{}-{}.png'.format(*pair_info)))

    # find a threshold
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(prediction, y_test)
    from bisect import bisect
    ti = bisect(tpr, 0.99)
    thres = thresholds[ti+1]
    out = "{} {} {} {th:.4f} {tp:.4f} {fp:.4f} {true} {fake}".format(
        pair_idx, *pair_info, th=thres, tp=tpr[ti+1], fp=fpr[ti+1],
        true=n_true, fake=n_fake)
    with open(os.path.join(output_dir, 'info_{}-{}.txt'.format(*pair_info)), 'a') as f:
        f.write(out)
