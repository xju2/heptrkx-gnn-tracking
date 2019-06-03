#!/bin/bash
export XDG_RUNTIME_DIR=/global/homes/x/xju/atlas/junk
export QT_QPA_PLATFORM='offscreen'

FILENAME=/global/cscratch1/sd/xju/heptrkx/output_graphs/raw_pairs/evt6011/pair000.h5
BASEDIR=trained_results/cluster

CLUSTER_DIR=trained_results/cluster/cluster_features
select_pairs $FILENAME 0.3514 $CLUSTER_DIR/selected/evt6011 --batch-size 512 --config ${BASEDIR}/cluster_features.yaml 

ALL_DIR=trained_results/cluster
select_pairs $FILENAME 0.9510 $ALL_DIR/selected/evt6011 --batch-size 512  --config ${BASEDIR}/all_features.yaml

LOC_DIR=trained_results/cluster/location_features
select_pairs $FILENAME 0.5384 $LOC_DIR/selected/evt6011 --batch-size 512  --config ${BASEDIR}/location_features.yaml

