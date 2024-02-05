#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "dsnn_smac_1"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:3

source /opt/software/pc2/EB-SW/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate dsnn_smac

python /scratch/hpc-prf-intexml/leonahennig/DeepShift/pytorch/cifar10_original.py --arch resnet20 --shift-depth 10 --shift-type Q --epochs 100 --lr 0.01376535177340809 --momentum 0.6232249074330178 --optimizer adagrad --rounding deterministic --batch-size 184 --weight-bits 8 --weight-decay 0.003186370955560786 --activation-bits 5 11