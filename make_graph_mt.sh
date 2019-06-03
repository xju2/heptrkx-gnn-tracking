#!/bin/bash

export XDG_RUNTIME_DIR=/global/homes/x/xju/atlas/junk
export QT_QPA_PLATFORM='offscreen'

make_graphs_from_pair_NNs \
	/global/homes/x/xju/atlas/heptrkx/trackml_inputs/train_all \
	/global/homes/x/xju/atlas/heptrkx/trackml_inputs/blacklist \
	6600 trained_results/doublets  \
	trained_results/doublets/merged_threshold.csv /global/cscratch1/sd/xju/heptrkx/output_graphs
