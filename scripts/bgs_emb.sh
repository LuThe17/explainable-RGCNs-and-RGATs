#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --export=NONE

python aifb_emb.py