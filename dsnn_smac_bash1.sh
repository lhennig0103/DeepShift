#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "dsnn_smac_1"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:3

conda activate dsnn_smac
python /scratch/hpc-prf-intexml/leonahennig/DeepShift/pytorch/smac_dsnn_mf.py