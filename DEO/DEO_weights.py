from functools import partial

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision.ops.misc import Permute

from utils.misc import load_pretrained_weights


class DEO(nn.Module):

    def __init__(self, model, path, device) -> None:
        super().__init__()

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
            features = self.forward_swin(x)

        return features
