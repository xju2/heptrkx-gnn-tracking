#!/bin/bash

for FILE in input_pairs/pair*
do
	echo $FILE
	python tf_train_pairs.py $FILE --batch-size 512 --epochs 6
done
