#!/bin/bash
#SBATCH -J hvd
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -A m1759


nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
#read -d " " -a array <<< "$nodes"

server=""
for ele in $nodes
do
	if [[ x$server == "x" ]]; then
		server=${ele}:8
	else
		server=${server},${ele}:8
	fi
done
echo $server

#mpirun -np 16 \
#	-H $server \
#	-bind-to none -map-by slot \
#	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#	-mca pml ob1 -mca btl ^openib \
#	python scripts/hvd_distributed
#
#srun --ntasks-per-node 8 -l python scripts/hvd_distributed

srun -l python scripts/hvd_distributed \
	--train-files "/global/cscratch1/sd/xju/heptrkx/kaggle/ReducedGraphSize/Embeded_doublets_padding/embeded_doublets_1*.tfrec" \
	--eval-files "/global/cscratch1/sd/xju/heptrkx/kaggle/ReducedGraphSize/Embeded_doublets_padding/embeded_doublets_2*.tfrec" \
	--job-dir "/global/cscratch1/sd/xju/heptrkx/kaggle/HVD/Test1" \
	--train-batch-size 1 \
	--eval-batch-size 1 
