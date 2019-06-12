#!/bin/bash
#SBATCH -J doublets
#SBATCH -C haswell
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -A m3253
#SBATCH --nodes 100

setup_heptrkx

which python
srun -n 100 -c 18 make_doublets_from_NNs configs/data_5000evts.yaml --workers 9

