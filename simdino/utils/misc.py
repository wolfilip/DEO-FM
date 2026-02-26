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
"""
Checkpoint and weight loading utilities.
"""

import argparse
import datetime
import math
import os
import subprocess
import time
import warnings
from collections import defaultdict, deque
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.ops.misc import Permute


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.

    The inputs corresponding to a single resolution are clubbed and a single
    forward pass is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.

    Supports both Vision Transformer (ViT) and Swin architectures with
    architecture-specific handling for multispectral and RGB processing.
    """

    def __init__(
        self,
        backbone,
        head_ms=None,
        head_rgb=None,
        cls_head=None,
        patch_head_late=None,
        patch_head_mid=None,
        to_distill=None,
    ):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head_ms = head_ms
        self.head_rgb = head_rgb
        self.to_distill = to_distill

        self.is_swin = self._is_swin_architecture()

        if self.is_swin:
            self._init_swin()
        else:
            self._init_vit()

        if to_distill == "student":
            self.cls_head = cls_head
            self.patch_head_late = patch_head_late
            self.patch_head_mid = patch_head_mid

    def _is_swin_architecture(self) -> bool:
        """Check if backbone is Swin-based (has features) vs ViT (no features)."""
        return hasattr(self.backbone, "features")

    def _init_swin(self):
        """Initialize Swin-specific components."""
        # Conv layers for Swin
        if self.to_distill == "student":
            norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
            norm_layer_rgb = partial(nn.LayerNorm, eps=1e-5)
            self.conv_ms = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.backbone.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_ms(self.backbone.features[0][0].norm1.normalized_shape[0]),
            )
            self.conv_rgb = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.backbone.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_rgb(self.backbone.features[0][0].norm1.normalized_shape[0]),
            )
        elif self.to_distill == "teacher":
            norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
            self.conv_ms = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.backbone.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_ms(self.backbone.features[0][0].norm1.normalized_shape[0]),
            )

    def _init_vit(self):
        """Initialize ViT-specific components."""
        # Patch embeddings for ViT
        if self.to_distill == "student":
            self.patch_embed_ms = PatchEmbed(
                in_chans=10,
                embed_dim=self.backbone.embed_dim,
            )
            self.patch_embed_rgb = PatchEmbed(
                in_chans=3,
                embed_dim=self.backbone.embed_dim,
            )
        elif self.to_distill == "teacher":
            self.patch_embed_ms = PatchEmbed(
                in_chans=10,
                embed_dim=self.backbone.embed_dim,
            )

    def _prepare_tokens(self, x):
        B, nc, w, h = x.shape
        if nc == 3:
            x = self.patch_embed_rgb(x)  # patch linear embedding
        else:
            x = self.patch_embed_ms(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        return self.backbone.pos_drop(x)

    def forward(self, x):
        if self.is_swin:
            return self._forward_swin(x)
        else:
            return self._forward_vit(x)

    def _forward_swin(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0

        if self.to_distill == "student":
            output_list = []
            rgb_cls_list = []
            for end_idx in idx_crops:
                ms_data = torch.cat(x[start_idx:end_idx])
                rgb_data = ms_data[:, :3, ...].clone()

                # MS data pass
                ms_data = self.conv_ms(ms_data)
                ms_data = self.backbone(ms_data)
                output_list.append(ms_data)

                del ms_data

                # RGB data pass
                rgb_data = self.conv_rgb(rgb_data)
                for i, layer in enumerate(self.backbone.features):
                    rgb_data = layer(rgb_data)
                    if end_idx == 2:
                        if i == 4:
                            rgb_patches_mid = rgb_data
                        if i == 6:
                            rgb_patches_late = rgb_data
                rgb_data = self.backbone.norm(rgb_data)
                rgb_data = self.backbone.permute(rgb_data)
                rgb_data = self.backbone.avgpool(rgb_data)
                rgb_data = self.backbone.flatten(rgb_data)
                rgb_data = self.backbone.head(rgb_data)
                rgb_cls_list.append(rgb_data)

                del rgb_data

                start_idx = end_idx

            output = torch.cat(output_list, dim=0)
            rgb_cls = torch.cat(rgb_cls_list, dim=0)

            del output_list, rgb_cls_list

            return (
                self.head_ms(output),
                self.cls_head(rgb_cls),
                self.patch_head_mid(rgb_patches_mid),
                self.patch_head_late(rgb_patches_late),
            )
        elif self.to_distill == "distiller":
            # distiller pass
            output_cls_list = []
            output_patch_list_mid = []
            output_patch_list_late = []
            for end_idx in idx_crops:
                _out_cls, _out_patch = self.backbone(torch.cat(x[start_idx:end_idx]))
                output_cls_list.append(_out_cls)
                output_patch_list_mid.append(_out_patch[0])
                output_patch_list_late.append(_out_patch[1])

                del _out_cls, _out_patch

                start_idx = end_idx

            output_cls = torch.cat(output_cls_list, dim=0)
            output_patch_mid = torch.cat(output_patch_list_mid, dim=0)
            output_patch_late = torch.cat(output_patch_list_late, dim=0)

            del output_cls_list, output_patch_list_mid, output_patch_list_late

            return output_cls, output_patch_mid, output_patch_late
        else:
            # teacher pass
            output_list = []
            for end_idx in idx_crops:
                ms_data = self.conv_ms(torch.cat(x[start_idx:end_idx]))
                _out = self.backbone(ms_data)
                output_list.append(_out)

                del _out

                start_idx = end_idx

            output = torch.cat(output_list, dim=0)

            del output_list

            return self.head_ms(output)

    def _forward_vit(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0

        if self.to_distill == "student":
            output_list = []
            rgb_cls_list = []
            for end_idx in idx_crops:
                ms_data = torch.cat(x[start_idx:end_idx])
                rgb_data = ms_data[:, :3, ...].clone()

                # MS data pass
                ms_data = self._prepare_tokens(ms_data)
                ms_data = self.backbone(ms_data)
                output_list.append(ms_data)

                del ms_data

                # RGB data pass
                rgb_data = self._prepare_tokens(rgb_data)
                if end_idx == 2:
                    rgb_patches = self.backbone.get_intermediate_layers(
                        rgb_data, (5, 11)
                    )
                    rgb_data = self.backbone.head(
                        self.backbone.norm(rgb_patches[-1][:, 0])
                    )
                else:
                    rgb_data = self.backbone(rgb_data)

                rgb_cls_list.append(rgb_data)

                del rgb_data

                start_idx = end_idx

            output = torch.cat(output_list, dim=0)
            rgb_cls = torch.cat(rgb_cls_list, dim=0)

            del output_list, rgb_cls_list

            return (
                self.head_ms(output),
                self.cls_head(rgb_cls),
                self.patch_head_late(rgb_patches[-1][:, 1:]),
                self.patch_head_mid(rgb_patches[0][:, 1:]),
            )
        elif self.to_distill == "distiller":
            # distiller pass
            output_cls_list = []
            output_patch_list_mid = []
            output_patch_list_late = []
            for end_idx in idx_crops:
                _out_cls, _out_patch = self.backbone(torch.cat(x[start_idx:end_idx]))
                output_cls_list.append(_out_cls)
                output_patch_list_mid.append(_out_patch[0])
                output_patch_list_late.append(_out_patch[1])

                del _out_cls, _out_patch

                start_idx = end_idx

            output_cls = torch.cat(output_cls_list, dim=0)
            output_patch_mid = torch.cat(output_patch_list_mid, dim=0)
            output_patch_late = torch.cat(output_patch_list_late, dim=0)

            del output_cls_list, output_patch_list_mid, output_patch_list_late

            return output_cls, output_patch_mid, output_patch_late
        else:
            # teacher pass
            output_list = []
            for end_idx in idx_crops:
                ms_data = torch.cat(x[start_idx:end_idx])
                ms_data = self._prepare_tokens(ms_data)
                _out = self.backbone(ms_data)
                output_list.append(_out)

                del _out

                start_idx = end_idx

            output = torch.cat(output_list, dim=0)

            del output_list

            return self.head_ms(output)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor # type: ignore
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g["weight_decay"])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def load_pretrained_weights(
    model, pretrained_weights, checkpoint_key, model_name, patch_size
):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `_orig_mod.` prefix induced by multicrop wrapper
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = (
                "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
            )
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # torchrun / elastic sets these
    # before existing checks add:
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(
            os.environ.get(
                "OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", 0)
            )
        )
        args.distributed = True
    elif (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    ):
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # fallback to SLURM (srun)
    elif (
        "SLURM_PROCID" in os.environ
        and "SLURM_NTASKS" in os.environ
        and "SLURM_LOCALID" in os.environ
    ):
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        args.gpu = int(os.environ["SLURM_LOCALID"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)

    # prefer env:// so torchrun / srun + MASTER_ADDR works
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message
