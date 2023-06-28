#!/bin/bash
#
#SBATCH -o /users/anddon76/icenet/icenet-gan/%j.out
#SBATCH -e /users/anddon76/icenet/icenet-gan/%j.err
#SBATCH -D /users/anddon76/icenet/icenet-gan/
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=gpu
#SBATCH --nodelist=node021
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb

cd /data/hpcdata/users/anddon76/icenet/icenet-gan/
mamba activate icenet-gan
python -m src.train_icenet --batch_size=2