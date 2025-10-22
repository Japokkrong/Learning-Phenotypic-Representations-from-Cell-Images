#!/bin/bash
#SBATCH --job-name=weak_dino
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/weak_dino_%j.out
#SBATCH --error=logs/weak_dino_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/ravipas.aph/warissara/WS-DINO/conda_envs/warissara_wsdino

cd ~/warissara/WS-DINO

MASTER_ADDR=localhost MASTER_PORT=29501 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python WS-DINO_BBBC021.py \
  --data_path BBBC021_annotated_fixed.csv \
  --epochs 200 \
  --output_dir ./checkpoints/weak_treatment
