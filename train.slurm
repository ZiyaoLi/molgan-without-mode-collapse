#!/bin/bash
#SBATCH -J mol050
#SBATCH -o ./mol050.stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH -t 5-00:00:00

python train_with_params.py -n mol050 -m molgan -r 5 --lam 0.5