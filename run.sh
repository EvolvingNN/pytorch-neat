#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 10G

python classification.py