# DINOv3 for BBBC021

## Overview
This script (`DINOv3_BBBC021.py`) implements fine-tuning of pretrained **DINOv3** models from Meta AI on the BBBC021 dataset, matching the experimental setup of the original DINO training for fair comparison.

## Key Features

### Pretrained Models (via timm)
Uses pretrained DINOv3 Vision Transformers (trained on 1.689B images):
- **vit_small_patch16_dinov3.lvd1689m** (default, ~22M params)
- **vit_base_patch16_dinov3.lvd1689m** (~86M params)
- **vit_large_patch16_dinov3.lvd1689m** (~304M params)
- **vit_giant_patch16_dinov3.lvd1689m** (~1.1B params)

### Training Setup
Identical to `DINO_BBBC021.py`:
- Same data augmentation pipeline
- Same learning rate schedule
- Same multi-crop strategy (2 global + 8 local crops)
- Same DINO loss function
- Same teacher-student EMA framework

## Usage

### Training
```bash
# Submit SLURM job
sbatch train_dinov3.slurm

# Or run directly
MASTER_ADDR=localhost MASTER_PORT=29502 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python DINOv3_BBBC021.py \
  --arch vit_small_patch16_dinov3.lvd1689m \
  --data_path BBBC021_annotated_fixed.csv \
  --epochs 200 \
  --output_dir ./checkpoints/dinov3_bbbc021
```

### Parameters
```python
--arch                    # DINOv3 model architecture (default: vit_small_patch16_dinov3.lvd1689m)
--patch_size             # Patch size (default: 16 for DINOv3)
--data_path              # Path to CSV with image annotations
--epochs                 # Number of training epochs (default: 400)
--batch_size_per_gpu     # Batch size per GPU (default: 16)
--lr                     # Learning rate (default: 0.000004)
--output_dir             # Directory to save checkpoints
```

## Comparison: DINO vs DINOv3

| Aspect | DINO (from scratch) | DINOv3 (pretrained) |
|--------|---------------------|---------------------|
| **Initialization** | Random or ImageNet weights | Pretrained on 1.689B diverse images |
| **Patch Size** | 8×8 | 16×16 |
| **Training Time** | Full training needed | Fine-tuning only |
| **Performance** | Learns from BBBC021 only | Leverages large-scale pretraining |
| **Embed Dim** | 384 (ViT-S/8) | 384 (ViT-S/16) |
| **Dataset Size** | ImageNet (1.2M) or none | 1.689B curated images |

## Expected Advantages of DINOv3

1. **Better initialization**: Trained on 1.689B curated images (vs 142M in DINOv2)
2. **Faster convergence**: Should require fewer epochs
3. **Better features**: State-of-the-art visual representations
4. **Transfer learning**: Benefits from massive diverse pretraining data

## Output

Checkpoints saved as:
- `DAPI_DINOv3_checkpoint.pth` (latest)
- `DAPI_DINOv3_checkpoint{epoch:04d}.pth` (periodic saves)

Training logs saved to:
- `DAPI_DINOv3_log.txt`

## Evaluation

Use the same evaluation pipeline:
```python
# In Eval_BBBC021.py, modify checkpoint path:
weights = f'./checkpoints/dinov3_bbbc021/DAPI_DINOv3_checkpoint{train_epoch:04d}.pth'
```

## Dependencies

Required packages (already in environment.yml):
- `timm==1.0.20` (for loading pretrained DINOv2)
- PyTorch, albumentations, pandas, etc.

## Notes

- DINOv3 uses **patch size 16** instead of 8
- Trained on **1.689 billion images** (latest and largest pretraining)
- Embedding dimension may differ based on model size
- The DINO head is randomly initialized and trained from scratch
- Only the backbone benefits from pretraining

## Model Sizes

```python
# Small (default) - ~22M params, 384D embeddings
--arch vit_small_patch16_dinov3.lvd1689m

# Base - ~86M params, 768D embeddings
--arch vit_base_patch16_dinov3.lvd1689m

# Large - ~304M params, 1024D embeddings
--arch vit_large_patch16_dinov3.lvd1689m

# Giant - ~1.1B params, 1536D embeddings (requires significant GPU memory!)
--arch vit_giant_patch16_dinov3.lvd1689m
```
