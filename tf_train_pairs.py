#!/usr/bin/env python3


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('pair_dir',  nargs='?', default='input_pairs/pair000.csv')
    add_arg('--batch-size',  type=int, default=64)
    add_arg('--epochs',  type=int, default=1)
    args = parser.parse_args()


    import tensorflow as tf
    from tensorflow import keras

    ## save checkpoints
    pairs_base_name = os.path.basename(args.pair_dir)
    checkpoint_path = "trained_results/doublets/model{}".format(pairs_base_name.replace('csv', 'ckpt'))
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
    df_input = pd.read_csv(args.pair_dir)

    all_inputs = df_input[['dphi', 'dz', 'dr', 'phi_slope', 'z0', 'deta', 'deta1', 'dphi1']].values
    all_targets = df_input[['true']].values
    n_total = all_inputs.shape[0]
    print("All Entries:", n_total)

    n_training = int(n_total*0.8)
    n_testing  = int(n_total*0.1)

    inputs = all_inputs[:n_training, :]
    targets = all_targets[:n_training, :]

    x_val = all_inputs[n_training:n_training+n_testing, :]
    y_val = all_targets[n_training:n_training+n_testing, :]

    x_test = all_inputs[n_training+n_testing:, :]
    y_test = all_targets[n_training+n_testing:, :]

    history = model.fit(inputs, targets,
                        epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(x_val, y_val),
                        callbacks = [cp_callback],
                        verbose=1)

    results = model.evaluate(x_test, y_test)
    print(results)

