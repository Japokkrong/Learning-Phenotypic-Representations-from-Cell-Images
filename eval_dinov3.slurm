#!/bin/bash
#SBATCH --job-name=eval_dinov3
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_dinov3_%j.out
#SBATCH --error=logs/eval_dinov3_%j.err

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/ravipas.aph/warissara/WS-DINO/conda_envs/warissara_wsdino

cd ~/warissara/WS-DINO

# Run evaluation for DINOv3 checkpoints
# Note: You need to modify Eval_BBBC021.py line 521 to point to DINOv3 checkpoints
# weights = f'./checkpoints/dinov3_bbbc021/DAPI_DINOv3_checkpoint{train_epoch:04d}.pth'

MASTER_ADDR=localhost MASTER_PORT=29503 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python Eval_BBBC021.py \
  --arch vit_small_patch16_dinov3 \
  --patch_size 16 \
  --batch_size_per_gpu 1 \
  --num_workers 1 \
  --checkpoint_key teacher \
  --dump_features features_dinov3
