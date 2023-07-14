#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --export=NONE
#SBATCH --mem=180000mb

python aifb_emb.py