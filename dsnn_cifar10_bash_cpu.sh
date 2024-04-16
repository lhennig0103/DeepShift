#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "great calculation"
#SBATCH -p normal

source /opt/software/pc2/EB-SW/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate dsnn_smac

srun python /scratch/hpc-prf-intexml/leonahennig/DeepShift/pytorch/cifar10_cpu.py --arch resnet20