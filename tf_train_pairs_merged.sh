#!/bin/bash

export XDG_RUNTIME_DIR=/global/homes/x/xju/atlas/junk
export QT_QPA_PLATFORM='offscreen'

for ((i=0; i < 90; i++))
do
	echo $i
	python tf_train_pairs.py configs/data.yaml $i
done
