#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=gpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

nvidia-smi
mamba activate icenet-gan
cd /users/anddon76/icenet/icenet-gan
python -m src.train_icenet --model=gan --generator_lambda=10