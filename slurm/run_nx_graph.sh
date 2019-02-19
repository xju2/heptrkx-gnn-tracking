#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres gpu:1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH -A atlas

which python
cd /global/homes/x/xju/track/gnn/code/gnn_networkx
srun python run_nx_graph.py
