# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import builtins
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import vision_transformer as vits
from carbontracker.tracker import CarbonTracker
from distillation_features import RADIO, DINOvX
from torchvision import models as torchvision_models
from utils.augmentations import DataAugmentationDINO, DataAugmentationDINOMS
from utils.datasets import fMoWMSDataset, fMoWRGBDataset
from utils.losses import DistLoss, MCRLoss, cosine_patch_loss
from utils.misc import (
    LARS,
    MetricLogger,
    MultiCropWrapper,
    bool_flag,
    cancel_gradients_last_layer,
    clip_gradients,
    cosine_scheduler,
    fix_random_seeds,
    get_params_groups,
    get_sha,
    get_world_size,
    has_batchnorms,
    init_distributed_mode,
    is_main_process,
    load_pretrained_weights,
    save_on_master,
)
from vision_transformer import DINOHead

import wandb

# For Swin to work properly with torchdynamo
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 32

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

EPS = 1e-6  # Small constant for numerical stability
REFERENCE_BATCH_SIZE = 256
GLOBAL_CROP_SIZE = 224
LOCAL_CROP_SIZE = 96
RGB_CHANNELS = 3
MS_CHANNELS = 7
DISTILLATION_ARCHS = {"dinov2", "dinov3", "radio"}
SWIN_STAGE_MID = 4
SWIN_STAGE_LATE = 6


def is_distillation_arch(arch: str) -> bool:
    return arch in DISTILLATION_ARCHS


def get_swin_stage_embed_dim(model: nn.Module, stage_idx: int) -> int:
    return int(model.features[stage_idx][1].norm1.normalized_shape[0])


def get_distill_bottleneck_dim(dist_arch_size: str) -> int:
    if dist_arch_size in {"small", "small_reg", "base", "base_reg"}:
        return 768
    if dist_arch_size in {"large", "large_reg"}:
        return 1024
    return 1024  # TODO: set a default for unknown sizes


torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "xcit",
            "deit_tiny",
            "deit_small",
        ]
        + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )
    parser.add_argument(
        "--z_dim",
        default=256,
        type=int,
        help="""Dimensionality of the DINO head bottleneck dim (default: 256).""",
    )
    parser.add_argument(
        "--hidden_dim",
        default=2048,
        type=int,
        help="""Dimensionality of the DINO head hidden dim (default: 2048).""",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--compile",
        type=bool_flag,
        default=True,
        help="""Whether or not compile model.""",
    )
    parser.add_argument(
        "--use_fp16",
        type=bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="stochastic depth rate"
    )

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.4, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="""Coefficient for the distillation loss.""",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Specify path to the training data.",
    )
    parser.add_argument(
        "--data_split",
        default=100000,
        type=int,
        help="How much data instances to use for training.",
    )
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Path to save logs and checkpoints."
    )
    parser.add_argument(
        "--saveckp_freq", default=50, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument("--nowandb", action="store_false", help="do not log wandb")
    parser.add_argument(
        "--dist_arch",
        default="",
        type=str,
        choices=["dinov2", "dinov3", "radio"],
        help="Specify distillation backbone.",
    )
    parser.add_argument(
        "--dist_arch_size",
        default="",
        type=str,
        choices=["small", "small_reg", "base", "base_reg", "large"],
        help="Specify distillation backbone size.",
    )
    parser.add_argument(
        "--init_student",
        action="store_true",
        help="initialize student with pretrained weights",
    )
    parser.add_argument(
        "--ms_arch",
        action="store_true",
        help="use multispectral architecture",
    )

    # MCR
    parser.add_argument(
        "--coeff",
        type=float,
        default=1,
        help="coefficient of cosine similarity (default: 1)",
    )
    parser.add_argument(
        "--eps", type=float, default=0.5, help="eps for TCR (default: 0.5)"
    )
    parser.add_argument(
        "--reduce_cov",
        type=int,
        default=0,
        help="""Whether or not all_reduce covariance matrices across gpus.""",
    )
    parser.add_argument(
        "--expa_type",
        type=int,
        default=1,
        help="""Whether or not apply smoothing in expansion_term.""",
    )
    return parser


def train_dino(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    cudnn.benchmark = True
    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir, "args.log"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
            f.write(str(vars(args)))
        import shutil

        shutil.copyfile("simdino/main_dino.py", f"{args.output_dir}/main.py")
    print_ = builtins.print
    log_file = Path(args.output_dir, "output.log")

    def print(*args, **kwargs):
        print_(*args, **kwargs)
        with open(log_file, "a") as f:
            if "file" in kwargs:
                del kwargs["file"]
            print_(*args, **kwargs, file=f)

    builtins.print = print
    if is_main_process():
        print("git:\n  {}\n".format(get_sha()))
        print(
            "\n".join(
                "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
            )
        )

    # ============ setup wandb ============
    if args.nowandb and is_main_process():
        runname = args.output_dir.rstrip("/").split("/")[-1]
        wandb.init(project="simdino", name=runname)
        wandb.config.update(args)

    # ============ carbontracker setup ============
    tracker = CarbonTracker(epochs=args.epochs)

    # ============ preparing data ... ============
    if args.ms_arch:
        transform = DataAugmentationDINOMS(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
    else:
        transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.dist_arch,
            args.ms_arch,
        )

    if args.ms_arch:
        dataset = fMoWMSDataset(args.data_path, args.data_split, transform=transform)
    else:
        dataset = fMoWRGBDataset(args.data_path, args.data_split, transform=transform)

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if is_main_process():
        print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
        if args.init_student:
            load_pretrained_weights(
                model=student,
                pretrained_weights="none",
                checkpoint_key="student",
                model_name=args.arch,
                patch_size=16,
            )
        if args.ms_arch:
            del student.patch_embed
            del teacher.patch_embed
    # if network is not a ViT (e.g., Swin)
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.norm.normalized_shape[0]
        # for Swin, we delete the patch embedding layer and replace it with separate patch embeddings for RGB and multispectral channels in the DINOHead
        if args.ms_arch:
            del student.features[0]
            del teacher.features[0]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions

    # we define distillation heads for class and patch tokens
    if is_distillation_arch(args.dist_arch):
        distill_head_cls = DINOHead(
            in_dim=(
                get_swin_stage_embed_dim(student, SWIN_STAGE_LATE)
                if "swin" in args.arch
                else embed_dim
            ),
            use_bn=args.use_bn_in_head,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=get_distill_bottleneck_dim(args.dist_arch_size),
        )
        distill_head_patch_late = DINOHead(
            in_dim=(
                get_swin_stage_embed_dim(student, SWIN_STAGE_LATE)
                if "swin" in args.arch
                else embed_dim
            ),
            use_bn=args.use_bn_in_head,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=get_distill_bottleneck_dim(args.dist_arch_size),
        )
        distill_head_patch_mid = DINOHead(
            in_dim=(
                get_swin_stage_embed_dim(student, SWIN_STAGE_MID)
                if "swin" in args.arch
                else embed_dim
            ),
            use_bn=args.use_bn_in_head,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=get_distill_bottleneck_dim(args.dist_arch_size),
        )

    student = MultiCropWrapper(
        backbone=student,
        head_ms=DINOHead(
            in_dim=embed_dim,
            use_bn=args.use_bn_in_head,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.z_dim,
        ),
        cls_head=distill_head_cls if is_distillation_arch(args.dist_arch) else None,
        patch_head_late=(
            distill_head_patch_late if is_distillation_arch(args.dist_arch) else None
        ),
        patch_head_mid=(
            distill_head_patch_mid if is_distillation_arch(args.dist_arch) else None
        ),
        to_distill="student",
    )
    teacher = MultiCropWrapper(
        backbone=teacher,
        head_ms=DINOHead(
            in_dim=embed_dim,
            use_bn=args.use_bn_in_head,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.z_dim,
        ),
        to_distill="teacher",
    )

    if args.dist_arch in {"dinov2", "dinov3"}:
        distillation_backbone = DINOvX(args)

        distillation_features = MultiCropWrapper(
            backbone=distillation_backbone,
            to_distill="distiller",
        )
    elif args.dist_arch == "radio":
        distillation_backbone = RADIO()

        distillation_features = MultiCropWrapper(
            backbone=distillation_backbone,
            to_distill="distiller",
        )
    else:
        distillation_features = None

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    if args.dist_arch:
        distillation_features = distillation_features.cuda()

    # synchronize batch norms (if any)
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        if args.dist_arch:
            distillation_features = nn.SyncBatchNorm.convert_sync_batchnorm(
                distillation_features
            )

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        distillation_features = nn.parallel.DistributedDataParallel(
            distillation_features, device_ids=[args.gpu]
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # teacher and student start with the same weights
    if not args.init_student:
        teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)

    # if and how to compile the model (with torchdynamo)
    if args.compile:
        if "swin" in args.arch:
            teacher = torch.compile(teacher, dynamic=False)
            student = torch.compile(student, dynamic=False)
        else:
            student = torch.compile(student)
            teacher = torch.compile(teacher)
        if args.dist_arch:
            if "swin" in args.dist_arch:
                distillation_features = torch.compile(
                    distillation_features, dynamic=False
                )
            else:
                distillation_features = torch.compile(distillation_features)

    # there is no backpropagation through the teacher (or the distiller), so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    if args.dist_arch:
        for p in distillation_features.parameters():
            p.requires_grad = False
    if is_main_process():
        print(f"Student and Teacher are built: they are both {args.arch} network.")
        if args.dist_arch:
            print(f"Distillation features are built: it is a {args.dist_arch} network.")

    print(
        f"Number of trainable parameters in student: {sum(p.numel() for p in student.parameters() if p.requires_grad)}"
    )

    # ============ preparing loss ... ============
    dino_loss = MCRLoss(
        args.local_crops_number
        + 2,  # total number of crops = 2 global crops + local_crops_number
        args.reduce_cov,
        args.expa_type,
        args.eps,
        args.coeff,
    ).cuda()

    # distillation loss
    if args.dist_arch:
        distillation_loss_cos = DistLoss(
            args.local_crops_number + 2, args.dist_arch
        ).cuda()
    else:
        distillation_loss_cos = None

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params_groups, fused=True
        )  # to use with ViTs and Swin
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9, fused=True
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = LARS(
            params_groups, fused=True
        )  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * get_world_size())
        / REFERENCE_BATCH_SIZE,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    if is_main_process():
        print("Loss, optimizer, and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth"),
    #     run_variables=to_restore,
    #     student=student,
    #     teacher=teacher,
    #     optimizer=optimizer,
    #     fp16_scaler=fp16_scaler,
    #     dino_loss=dino_loss,
    # )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    if is_main_process():
        print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        tracker.epoch_start()
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
            distiller=distillation_features,
            distiller_loss_cos=distillation_loss_cos,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }
        if args.dist_arch:
            save_dict["distiller"] = distillation_features.state_dict()
            save_dict["distillation_loss_cos"] = distillation_loss_cos.state_dict()
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        tracker.epoch_end()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_main_process():
        print("Training time {}".format(total_time_str))

    tracker.stop()


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
    distiller=None,
    distiller_loss_cos=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, (images, image_ms) in enumerate(
        metric_logger.log_every(data_loader, 200, header)
    ):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        if args.dist_arch == "ms":
            image_ms = image_ms.cuda(non_blocking=True)

        # split images into rgb and multispectral channels
        if is_distillation_arch(args.dist_arch) and args.ms_arch:
            images_rgb = [
                torch.split(image, [RGB_CHANNELS, MS_CHANNELS], dim=1)[0]
                for image in images[:2]
            ]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output_ms = teacher(images[:2])

            if distiller is not None:
                (
                    distillation_output_cls,
                    distillation_output_patch_mid,
                    distillation_output_patch_late,
                ) = distiller(images_rgb[:2])

            (
                student_output,
                student_distill_cls,
                student_distill_patch_mid,
                student_distill_patch_late,
            ) = student(images)

            loss, comp_loss, expa_loss = dino_loss(student_output, teacher_output_ms)

            if distiller is not None:
                if distillation_output_patch_late is not None:
                    distillation_loss_patch_late = cosine_patch_loss(
                        args.dist_arch,
                        student_distill_patch_late,
                        distillation_output_patch_late,
                    )
                    distillation_loss_patch_mid = cosine_patch_loss(
                        args.dist_arch,
                        student_distill_patch_mid,
                        distillation_output_patch_mid,
                    )
                    distillation_loss_patch_cos = (
                        distillation_loss_patch_late + distillation_loss_patch_mid
                    ) / 2
                    distillation_loss = -distillation_loss_patch_cos

                if distillation_output_cls is not None:
                    distillation_loss_cls_cos = distiller_loss_cos(
                        student_distill_cls, distillation_output_cls
                    )
                    distillation_loss = -distillation_loss_cls_cos

                if (
                    distillation_output_patch_late is not None
                    and distillation_output_cls is not None
                ):
                    distillation_loss = (
                        -(distillation_loss_patch_cos + distillation_loss_cls_cos) / 2
                    )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if distiller is not None:
            if not math.isfinite(distillation_loss.item()):
                print(
                    "Distillation Loss is {}, stopping training".format(
                        distillation_loss.item()
                    ),
                    force=True,
                )
                sys.exit(1)

            total_loss = loss + distillation_loss
        else:
            total_loss = loss

        # ============ backward and optim step ... ============

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            total_loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(total_loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher (modified to be more customizble)
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            # for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
            #     param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if hasattr(torch, "_foreach_lerp_"):
                teacher_params = list(teacher_without_ddp.parameters())
                student_params = list(student.module.parameters())
                for teacher_param, student_param in zip(teacher_params, student_params):
                    teacher_param.data.mul_(m).add_(
                        (1 - m) * student_param.detach().data
                    )
                # torch._foreach_lerp_(
                #     list(teacher_without_ddp.parameters()),
                #     list(student.module.parameters()),
                #     weight=1.0 - m,
                # )
            else:
                torch._foreach_mul_(list(teacher_without_ddp.parameters()), m)
                torch._foreach_add_(
                    list(teacher_without_ddp.parameters()),
                    list(student.module.parameters()),
                    alpha=1.0 - m,
                )

        # logging
        torch.cuda.synchronize()
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(expa_loss=expa_loss)
        metric_logger.update(comp_loss=comp_loss)
        if distiller is not None:
            if distillation_loss_cls_cos is not None:
                metric_logger.update(
                    distillation_loss_cls_cos=distillation_loss_cls_cos
                )
            if distillation_loss_patch_cos is not None:
                metric_logger.update(
                    distillation_loss_patch_cos=distillation_loss_patch_cos
                )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if is_main_process() and args.nowandb:
            logs2wb = {
                "loss": total_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "wd": optimizer.param_groups[0]["weight_decay"],
            }
            logs2wb.update(
                {
                    "expa_loss": expa_loss,
                    "comp_loss": comp_loss,
                }
            )
            if distiller is not None:
                logs2wb.update({"dist_loss_cls_cos": distillation_loss_cls_cos})
                logs2wb.update({"dist_loss_patch_cos": distillation_loss_patch_cos})
            wandb.log(logs2wb)
    metric_logger.synchronize_between_processes()
    if is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    train_dino(args)
