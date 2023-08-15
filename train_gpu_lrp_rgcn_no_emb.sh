#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --export=NONE
#SBATCH --mem-per-gpu=376000

python train_gpu_lrp_rgcn_no_emb.py