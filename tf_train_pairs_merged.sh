#!/bin/bash

All_Pairs=/global/cscratch1/sd/xju/heptrkx/pairs/all_pairs/evt6600
True_Pairs=/global/cscratch1/sd/xju/heptrkx/pairs/merged_true_pairs/training
for FILE in $All_Pairs/*
do
	NAME=$(basename $FILE)
	PAIR=${NAME:4:3}
	#echo $FILE, $NAME, $PAIR, ${True_Pairs}/$NAME
	echo $PAIR
	python tf_train_pairs.py $FILE --batch-size 512 --epochs 2 --true-file ${True_Pairs}/$NAME --in-eval
done
