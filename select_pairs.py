#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Keras train pairs for each layer-pairs')
    add_arg = parser.add_argument
    add_arg('file_name', type=str, help='file name for pair candidates')
    add_arg('cut_on_score', type=float, help='selection criteria')
    add_arg('output_dir',   type=str, help='save selected pairs')
    add_arg('--batch-size', type=int, default=64)
    add_arg('--config',       type=str, default=None, help='config of training')
    add_arg('--model-weight', type=str, default=None, help='model weight')

    args = parser.parse_args()

    cut = args.cut_on_score
    file_name = args.file_name
    output_dir = args.output_dir
    # check if there is a config
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            config = yaml.load(f)
        pair_basename = os.path.basename(config['data']['file_name']).replace('.h5', '.ckpt')
        model_weight_dir = os.path.join(config['output_dir'], 'model{}'.format(pair_basename))
        features = config['data']['features']
    else:
        features = ['dphi', 'dz', 'dr', 'phi_slope', 'z0', 'deta', 'deta1', 'dphi1']

    if args.model_weight is not None:
        model_weight_dir = args.model_weight

    print("model weight:", model_weight_dir)
    print("Features:", features)
    print("output:", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    pairs_base_name = os.path.basename(file_name)
    outname = os.path.join(output_dir, pairs_base_name)
    if os.path.exists(outname):
        print(outname, 'is there')
        exit()

    import numpy as np
    import pandas as pd

    with pd.HDFStore(file_name) as store:
        df_input = store.get('data')

    from nx_graph.shadow_model import fully_connected_classifier
    model = fully_connected_classifier()
    model.load_weights(model_weight_dir)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    from tf_train_pairs import keep_finite
    df_input = keep_finite(df_input)
    print("Before selection:", df_input.shape)
    all_inputs = df_input[features].values
    all_inputs = scaler.fit_transform(all_inputs)
    prediction = model.predict(all_inputs, batch_size=args.batch_size)

    from nx_graph.utils_plot import plot_metrics
    from make_true_pairs_for_training_segments_mpi import layer_pairs_dict

    pair_idx = int(pairs_base_name[4:-3])
    pair_info = layer_pairs_dict[pair_idx]

    all_targets = df_input[['true']].values
    plot_metrics(prediction, all_targets,
                 outname=os.path.join(output_dir, 'roc{}_{}-{}.png'.format(pair_idx, *pair_info)))

    df_input = df_input.assign(prediction=prediction, selected=(prediction > cut))


    with pd.HDFStore(outname) as store:
        store['data'] = df_input
