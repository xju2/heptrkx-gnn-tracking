#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=xju@lbl.gov
#SBATCH -N 4

. scripts/setup.sh
#srun -n 128 -l python ./train.py -d configs/segclf.yaml
srun -n 128 -l python ./train.py -d configs/hello_test.yaml
