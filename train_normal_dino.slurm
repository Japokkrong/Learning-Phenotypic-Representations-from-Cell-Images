#!/bin/bash
#SBATCH --job-name=normal_dino
#SBATCH --partition=defq          # or whatever your cluster uses
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --cpus-per-task=8         # adjust based on your cluster
#SBATCH --mem=32G                 # memory allocation
#SBATCH --time=24:00:00           # max runtime (24h)
#SBATCH --output=logs/normal_dino_%j.out
#SBATCH --error=logs/normal_dino_%j.err

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/ravipas.aph/warissara/WS-DINO/conda_envs/warissara_wsdino

cd ~/warissara/WS-DINO

MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python DINO_BBBC021.py \
  --data_path BBBC021_annotated_fixed.csv \
  --epochs 200 \
  --output_dir ./checkpoints/normal_dino
