#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:0
#SBATCH --mem 10G

python mnist.py