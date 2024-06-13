#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --mincpu=20

srun python train.py optimizer.lr=0.001
