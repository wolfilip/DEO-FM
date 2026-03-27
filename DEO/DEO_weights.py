from functools import partial

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision.ops.misc import Permute
import vision_transformer as vits

from utils.misc import load_pretrained_weights


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DEO(nn.Module):

    def __init__(self, model, path, device) -> None:
        super().__init__()

        self.patch_size = 8
        self.model = model

        if "vit" in model:
            self.feat_extr = vits.__dict__[model](patch_size=self.patch_size)
            self.feat_extr.patch_embed_ms = PatchEmbed(
                img_size=256,
                patch_size=self.patch_size,
                in_chans=10,
                embed_dim=self.feat_extr.embed_dim,
            )
            self.feat_extr.patch_embed_rgb = PatchEmbed(
                img_size=256,
                patch_size=self.patch_size,
                in_chans=3,
                embed_dim=self.feat_extr.embed_dim,
            )
        elif "swin" in model:
            self.feat_extr = torchvision_models.__dict__[
                model
            ]()  # load backbone architecture
            del self.feat_extr.features[0]
            # Conv layers for Swin
            norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
            norm_layer_rgb = partial(nn.LayerNorm, eps=1e-5)
            self.feat_extr.conv_ms = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.feat_extr.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_ms(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
            )
            self.feat_extr.conv_rgb = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.feat_extr.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_rgb(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
            )

        load_pretrained_weights(self.feat_extr, path, "student", model)

        # put into eval mode and to device
        self.feat_extr.eval()
        self.feat_extr.to(device)

        for p in self.feat_extr.parameters():
            p.requires_grad = False

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        if nc == 3:
            x = self.feat_extr.patch_embed_rgb(x)  # patch linear embedding
        else:
            x = self.feat_extr.patch_embed_ms(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.feat_extr.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.feat_extr.interpolate_pos_encoding(x, w, h)

        return self.feat_extr.pos_drop(x)

    def forward_swin(self, x):

        features = []
        if x.shape[1] == 10:
            x = self.feat_extr.conv_ms(x)
        else:
            x = self.feat_extr.conv_rgb(x)

        # extarct intermediate swin layers
        for i, layer in enumerate(self.feat_extr.features):
            x = layer(x)
            if i in [0, 2, 4, 6]:
                features.append(x)

        return features

    def forward(self, x):
        with torch.no_grad():
            if "swin" in self.model:
                features = self.forward_swin(x)
            else:
                x = self.prepare_tokens(x)
                features = self.feat_extr.get_intermediate_layers(x=x, n=[3, 5, 8, 11])

        return features
