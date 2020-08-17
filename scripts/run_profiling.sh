#!/bin/bash
#SBATCH -J profiling
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

which python
train_nx_graph_single heptrkx/configs/config_embeded_doublets_profiling.yaml
