#!/bin/bash
#SBATCH -J mgan_qed
#SBATCH -o ./mgan_qed.stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH -t 5-00:00:00

python train_with_params -n pacstats_01 -m pacstats -r 1