#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "dsnn_smac_1"
#SBATCH -p gpu
#SBATCH --gpus-per-task=1

source /opt/software/pc2/EB-SW/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate dsnn_smac

srun python /scratch/hpc-prf-intexml/leonahennig/DeepShift/pytorch/cifar10.py --arch resnet20-