# DINOv3 fine-tuning for BBBC021 dataset
# Uses pretrained DINOv3 models from timm library
# Adapted to match the structure of DINO_BBBC021.py for fair comparison

import argparse
import os
import sys
import datetime
import time
import cv2
import math
import json
from pathlib import Path
sys.path.append('') # add path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pandas as pd
import albumentations 
from torch.utils.data import Dataset
import timm
import utils
from vision_transformer import DINOHead

def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small_patch16_dinov3.lvd1689m', type=str,
        help="""DINOv3 model architecture from timm. Options:
        - vit_small_patch16_dinov3.lvd1689m (ViT-S/16, ~22M params)
        - vit_base_patch16_dinov3.lvd1689m (ViT-B/16, ~86M params)
        - vit_large_patch16_dinov3.lvd1689m (ViT-L/16, ~304M params)
        - vit_giant_patch16_dinov3.lvd1689m (ViT-g/16, ~1.1B params)""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Patch size for DINOv3 models (typically 16).""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.""")
    parser.add_argument('--momentum_teacher', default=0.99, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.0, help="""Final value of the
        weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.000004, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=0.000003, help="""Target LR at the
        end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.2, 0.3),
        help="""Scale range of the cropped image before resizing for global views.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.15),
        help="""Scale range of the cropped image before resizing for local views.""")

    # Misc
    parser.add_argument('--data_path', default='./BBBC021_annotated.csv', type=str,      
        help='Path to the training data CSV.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--channel_headers', default=['Image_FileName_DAPI','Image_FileName_Tubulin', 'Image_FileName_Actin'], type=list)

    return parser


class NaturalImageDataset(Dataset):
    def __init__(self, path10, local_crops_number, args):
        print("Preparing NaturalImageDataset for DINOv3...")
        path0 = pd.read_csv(path10)
        
        self.X0 = path0[args.channel_headers[0]]  # DAPI channel
        
        # Augmentations for global crops (224x224)
        self.aug0 = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(
                (224, 224), scale=(0.1, 0.2), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=True),
        ])
        
        self.aug1 = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(
                (224, 224), scale=(0.1, 0.2), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=True),
        ])
        
        self.local_crops_number = local_crops_number
        
        # Augmentations for local crops (96x96)
        self.aug_local_crop = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(
                (96, 96), scale=(0.04, 0.08), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=True),
        ])
        
    def __len__(self):
        return len(self.X0)
        
    def __getitem__(self, i):
        # Load image
        Aimage = Image.open(self.X0[i])
        Aimage = np.array(Aimage)
        Aimage[Aimage > 10000] = 10000

        # Global crop 0
        transformed0 = self.aug0(image=Aimage)
        image_0 = transformed0['image'].astype(np.float32)

        # Global crop 1
        transformed1 = self.aug1(image=Aimage)
        image_1 = transformed1['image'].astype(np.float32)
  
        # Expand to 3 channels (convert grayscale to RGB)
        image_0 = np.expand_dims(image_0, 0)      
        image_1 = np.expand_dims(image_1, 0)
     
        image_0 = np.concatenate((image_0, image_0, image_0), axis=0)
        image_1 = np.concatenate((image_1, image_1, image_1), axis=0)
                
        image_0 = torch.tensor(image_0, dtype=torch.float)
        image_1 = torch.tensor(image_1, dtype=torch.float) 
        
        crops = []
        crops.append(image_0)
        crops.append(image_1)
        
        # Local crops
        for _ in range(self.local_crops_number):
            transformed2 = self.aug_local_crop(image=Aimage)
            image_2 = transformed2['image'].astype(np.float32)      
            image_2 = np.expand_dims(image_2, 0)                
            image_2 = np.concatenate((image_2, image_2, image_2), axis=0)
            image_2 = torch.tensor(image_2, dtype=torch.float) 
            crops.append(image_2)

        return crops


class DINOv3Wrapper(nn.Module):
    """Wrapper to add DINO head on top of pretrained DINOv3 backbone"""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        # DINOv2 returns features
        if isinstance(x, list):
            # Multi-crop forward
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            output = []
            for end_idx in idx_crops:
                _out = self.backbone.forward_features(torch.cat(x[start_idx: end_idx]))
                output.append(_out)
                start_idx = end_idx
            output = torch.cat(output)
        else:
            output = self.backbone.forward_features(x)
        
        # Apply DINO head
        return self.head(output)


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    dataset = NaturalImageDataset(args.data_path, local_crops_number=args.local_crops_number, args=args)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    print(f"Creating DINOv3 model: {args.arch}")
    
    # Load pretrained DINOv3 backbone from timm
    student_backbone = timm.create_model(args.arch, pretrained=True, num_classes=0)  # num_classes=0 for feature extraction
    teacher_backbone = timm.create_model(args.arch, pretrained=True, num_classes=0)
    
    # Get embedding dimension from the model
    embed_dim = student_backbone.num_features
    print(f"Embedding dimension: {embed_dim}")
    
    # Create DINO heads
    student_head = DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )
    teacher_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    
    # Wrap with multi-crop wrapper
    student = utils.MultiCropWrapper(student_backbone, student_head)
    teacher = utils.MultiCropWrapper(teacher_backbone, teacher_head)
    
    # Move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    
    # Synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # We need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    # Teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    
    # There is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # For mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # Momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINOv3 training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'DAPI_DINOv3_checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'DAPI_DINOv3_checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "DAPI_DINOv3_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        
        # Teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.07,
                 center_momentum=0.8):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # We apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # Ensure warmup epochs don't exceed total epochs
        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINOv3', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
