#!/bin/bash
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "dsnn_smac_1"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:3

source /opt/software/pc2/EB-SW/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate DeepShift_270525 

python /scratch/hpc-prf-intexml/leonahennig/DeepShift/pytorch/smac_dsnn_mf_short.py
