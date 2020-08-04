#!/bin/bash
#SBATCH -J hvd-gnn
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

DATADIR=/global/cscratch1/sd/xju/heptrkx/codalab
srun -l python scripts/hvd_distributed \
	--train-files "$DATADIR/Daniel_Doublets_RemoveDuplicatedHits_xyz_train/*_*.tfrec" \
	--eval-files "$DATADIR/Daniel_Doublets_RemoveDuplicatedHits_xyz_val/*_*.tfrec" \
	--job-dir "/global/cscratch1/sd/xju/heptrkx/kaggle/HVD/Test16" \
	--train-batch-size 1 \
	--eval-batch-size 1  \
	--num-epochs 3 \
	-d 
