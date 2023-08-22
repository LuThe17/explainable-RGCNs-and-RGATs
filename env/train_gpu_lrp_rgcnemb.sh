#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --export=NONE
#SBATCH --mem-per-gpu=376000



python main.py  --learning_rate 0.01 --datset_name 'AIFB' --homedir  '/pfs/work7/workspace/scratch/ma_luitheob-master/AIFB/' --epochs 100 --model_name 'RGCN_no_emb'