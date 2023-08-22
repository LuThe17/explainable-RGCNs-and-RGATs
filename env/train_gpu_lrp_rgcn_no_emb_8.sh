#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --export=NONE
#SBATCH --mem=752000

python train_gpu_lrp_rgcn_no_emb.py