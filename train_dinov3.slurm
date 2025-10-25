#!/bin/bash
#SBATCH --job-name=dinov3_bbbc021
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dinov3_%j.out
#SBATCH --error=logs/dinov3_%j.err

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/ravipas.aph/warissara/WS-DINO/conda_envs/warissara_wsdino

cd ~/warissara/DINOv3

# Install timm if not already installed
pip install timm --quiet

# Run DINOv3 training with same parameters as normal DINO
MASTER_ADDR=localhost MASTER_PORT=29502 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python DINOv3_BBBC021.py \
  --arch vit_small_patch16_dinov3.lvd1689m \
  --data_path BBBC021_annotated_fixed.csv \
  --epochs 200 \
  --output_dir ./checkpoints/dinov3_bbbc021
