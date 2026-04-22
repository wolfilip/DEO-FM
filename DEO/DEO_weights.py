from typing import List

from functools import partial

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision.ops.misc import Permute
import vision_transformer as vits

from utils.misc import load_pretrained_weights


class PatchEmbed(nn.Module):
    """Image-to-patch embedding layer.

    Converts an input image tensor into a sequence of patch tokens using a
    strided convolution.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an image tensor into patch tokens.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch tokens of shape (B, N, D).
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DEO(nn.Module):
    """Wrapper around a pretrained DEO backbone for feature extraction.

    The wrapper builds either a ViT- or Swin-based backbone, creates either
    RGB and multispectral patch embedding layers, loads pretrained weights, and
    exposes frozen intermediate features for downstream use.
    """

    def __init__(self, model: str, path: str, device: torch.device | str) -> None:
        """Initialize the frozen feature extractor.

        Args:
            model (str): Backbone name used to resolve the architecture.
            path (str): Path to the pretrained checkpoint.
            device (torch.device | str): Device used to place the backbone.
        """
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

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare ViT tokens for RGB or multispectral input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tokenized input with CLS and positional embeddings.
        """
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

    def forward_swin(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate Swin features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: Selected intermediate feature maps.
        """

        features: List[torch.Tensor] = []
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract frozen intermediate features from the configured backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where C is either 3 (RGB) or 10 (multispectral).

        Returns:
            List[torch.Tensor]: Intermediate features from the ViT or Swin
            backbone.
        """
        with torch.no_grad():
            if "swin" in self.model:
                features = self.forward_swin(x)
            else:
                x = self.prepare_tokens(x)
                features = self.feat_extr.get_intermediate_layers(x=x, n=[3, 5, 8, 11])

        return features
