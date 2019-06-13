#!/bin/bash
#SBATCH -J doublets
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A m3253
#SBATCH --nodes 64

setup_heptrkx

which python
srun -n 64 make_doublets_from_NNs configs/data_5000evts.yaml
