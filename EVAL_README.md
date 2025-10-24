# Evaluation SLURM Scripts

## Overview
These scripts run evaluation (NSC/NSCB metrics) on trained model checkpoints.

## Available Scripts

### 1. `eval_dinov3.slurm` - Evaluate DINOv3 Models
```bash
sbatch eval_dinov3.slurm
```
- Architecture: `vit_small_patch16_dinov3`
- Patch size: 16×16
- Evaluates: `./checkpoints/dinov3_bbbc021/DAPI_DINOv3_checkpoint*.pth`

### 2. `eval_dino.slurm` - Evaluate Normal DINO Models
```bash
sbatch eval_dino.slurm
```
- Architecture: `vit_small`
- Patch size: 8×8
- Evaluates: `./checkpoints/normal_dino/DAPI_DINO_checkpoint*.pth`

### 3. `eval_ws_dino.slurm` - Evaluate WS-DINO Models
```bash
sbatch eval_ws_dino.slurm
```
- Architecture: `vit_small`
- Patch size: 8×8
- Evaluates: `./checkpoints/weak_treatment/DAPI_weak_compound_DINO_checkpoint*.pth`

---

## Important: Edit Checkpoint Path Before Running

⚠️ **You MUST edit `Eval_BBBC021.py` line 521** to specify which checkpoints to evaluate:

### For DINOv3:
```python
# Line 521 in Eval_BBBC021.py
weights = f'./checkpoints/dinov3_bbbc021/DAPI_DINOv3_checkpoint{train_epoch:04d}.pth'
```

### For Normal DINO:
```python
# Line 521 in Eval_BBBC021.py
weights = f'./checkpoints/normal_dino/DAPI_DINO_checkpoint{train_epoch:04d}.pth'
```

### For WS-DINO:
```python
# Line 521 in Eval_BBBC021.py
weights = f'./checkpoints/weak_treatment/DAPI_weak_compound_DINO_checkpoint{train_epoch:04d}.pth'
```

---

## What the Scripts Do

1. **Load trained model** from checkpoints
2. **Extract features** from BBBC021 images (4 crops per image, median aggregation)
3. **Apply TVN correction** using DMSO controls
4. **Aggregate features** at treatment level
5. **Calculate NSC/NSCB metrics** (nearest neighbor matching by MoA)
6. **Save results**:
   - `aggregated_features_{channel}_epoch_{epoch}.csv`
   - `NSCB_aggregated_features_*`
   - Console output with accuracy scores

---

## Resource Requirements

- **GPU**: 1 GPU
- **CPUs**: 8 cores
- **Memory**: 32GB RAM
- **Time**: 12 hours max

---

## Output Logs

- Standard output: `logs/eval_{model}_{jobid}.out`
- Error output: `logs/eval_{model}_{jobid}.err`

---

## Epoch Range

Currently evaluates epochs: **0, 5, 10, 15, 20, 25, 30** (line 518 in Eval_BBBC021.py)

To change the range, edit:
```python
for train_epoch in range(0, 35, 5):  # Adjust this range
```

---

## Running All Evaluations

```bash
# Evaluate all models in sequence
sbatch eval_dino.slurm
sbatch eval_ws_dino.slurm  
sbatch eval_dinov3.slurm
```

Or run them in parallel (if you have enough resources):
```bash
sbatch eval_dino.slurm & sbatch eval_ws_dino.slurm & sbatch eval_dinov3.slurm
```

---

## Key Differences Between Models

| Model | Architecture | Patch Size | Pretraining | Expected Performance |
|-------|-------------|------------|-------------|---------------------|
| DINO | ViT-S/8 | 8×8 | ImageNet or scratch | Baseline |
| WS-DINO | ViT-S/8 | 8×8 | ImageNet + weak labels | Better than DINO |
| DINOv3 | ViT-S/16 | 16×16 | 1.689B images | Best (expected) |
