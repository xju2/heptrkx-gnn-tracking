#!/usr/bin/env python3


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('file_name',  nargs='?', default='input_pairs/pair000.h5', help='file name for pair candidates')
    add_arg('--batch-size',  type=int, default=64)
    add_arg('--epochs',  type=int, default=1)
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


    history = model.fit(inputs, targets,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks = [cp_callback],
                        class_weight={0: 0.1, 1: 100},
                        verbose=1)

    prediction = model.predict(x_test,
                               batch_size=batch_size)

    from nx_graph.utils_plot import plot_metrics
    plot_metrics(prediction, y_test,
                 outname='trained_results/doublets/roc_{}'.format(pairs_base_name.replace('h5', 'png')))

