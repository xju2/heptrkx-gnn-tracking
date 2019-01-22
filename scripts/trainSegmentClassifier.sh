#!/bin/bash
#SBATCH -J segclf-big
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=xju@lbl.gov
#SBATCH -N 4
#SBATCH -o logs/%x-%j.out


mkdir -p logs
. scripts/setup.sh

#srun -n 128 -l python ./train.py -d configs/segclf.yaml
srun -n 128 -l python ./train.py -d configs/hello_test.yaml


#srun -l python ./train.py -v -d configs/segclf_big.yaml

