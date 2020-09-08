#!/bin/bash
#SBATCH -J hvd-test
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out


ifconfig
srun -l python scripts/hvd_example
