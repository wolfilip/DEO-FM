from typing import Tuple

import torch
import torch.nn as nn

# Layer indices for intermediate feature extraction
# These layers are chosen for diverse feature quality based on the DINOv2/v3 architecture:
# - For small/base models: layers 8 and 11 capture mid-level and high-level features
# - For large models: layers 15 and 23 provide similar semantic feature levels
LAYER_INDICES_SMALL_BASE = (8, 11)
LAYER_INDICES_LARGE = (15, 23)
DEFAULT_LAYER_INDICES = (11,)  # Fallback for unrecognized model sizes


class DINOvX(nn.Module):
    """
    Wrapper for DINOv2 [https://arxiv.org/abs/2304.07193] and DINOv3 [https://arxiv.org/abs/2508.10104] feature extraction models.

    This class loads pretrained DINO models (v2 or v3) from torch.hub and extracts
    intermediate layer features suitable for knowledge distillation. It returns both
    class tokens and patch tokens from selected transformer layers.

    The extracted features are:
    - Class token (CLS): Global image representation from the final layer
    - Patch tokens: Spatial feature maps from intermediate layers

    Args:
        args: Argument namespace containing:
            - dist_arch (str): Architecture type, either "dinov2" or "dinov3"
            - dist_arch_size (str): Model size, one of "small", "small_reg", "base",
                                    "base_reg", "large", "large_reg"

    Attributes:
        dist_arch (str): The architecture type (dinov2/dinov3)
        dist_arch_size (str): The model size variant
        feat_extr (nn.Module): The loaded pretrained DINO model

    Example:
        >>> args = argparse.Namespace(dist_arch="dinov2", dist_arch_size="base")
        >>> model = DINOvX(args)
        >>> cls_token, patch_tokens = model(images)
    """

    def __init__(self, args) -> None:
        super().__init__()

        self.dist_arch_size = args.dist_arch_size
        self.dist_arch = args.dist_arch

        if args.dist_arch == "dinov2":
            if args.dist_arch_size == "small":
                self.feat_extr = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14"
                )
            elif args.dist_arch_size == "small_reg":
                self.feat_extr = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                )
            elif args.dist_arch_size == "base":
                self.feat_extr = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitb14"
                )
            elif args.dist_arch_size == "base_reg":
                self.feat_extr = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitb14_reg"
                )
            elif args.dist_arch_size == "large":
                self.feat_extr = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitl14"
                )

        # clone DINOv3 repo locally (https://github.com/facebookresearch/dinov3/tree/main) and ask for weights (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Paste the link into weights=""
        # Also possible to use Hugging Face Hub (https://huggingface.co/docs/transformers/en/model_doc/dinov3).
        elif args.dist_arch == "dinov3":
            if args.dist_arch_size == "large":
                self.feat_extr = torch.hub.load(
                    "../dinov3",
                    "dinov3_vitl16",
                    source="local",
                    weights="",
                )
            elif args.dist_arch_size == "base":
                self.feat_extr = torch.hub.load(
                    "../dinov3",
                    "dinov3_vitb16",
                    source="local",
                    weights="",
                )

        self.feat_extr.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Extract intermediate features from DINO model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where:
                - B: batch size
                - C: number of channels (typically 3 for RGB)
                - H, W: image height and width

        Returns:
            Tuple containing:
                - out_cls (torch.Tensor): Class token features from the last selected layer,
                                         shape (B, D) where D is the embedding dimension
                - out_patch (Tuple[torch.Tensor, ...]): Tuple of patch token features from
                                                        intermediate layers, each of shape
                                                        (B, N, D) where N is number of patches
        """
        with torch.no_grad():
            # Select appropriate layer indices based on model size
            # Smaller models have fewer layers, so we extract from earlier positions
            if (
                self.dist_arch_size == "small"
                or self.dist_arch_size == "small_reg"
                or self.dist_arch_size == "base"
                or self.dist_arch_size == "base_reg"
            ):
                layers = LAYER_INDICES_SMALL_BASE
            elif self.dist_arch_size == "large" or self.dist_arch_size == "large_reg":
                layers = LAYER_INDICES_LARGE
            else:
                layers = DEFAULT_LAYER_INDICES

            # Handle different API signatures between DINOv2 and DINOv3
            try:
                # DINOv2 API: positional arguments
                intermediate = self.feat_extr.get_intermediate_layers(
                    x, layers, return_class_token=True
                )
            except TypeError:
                # DINOv3 API: keyword arguments
                intermediate = self.feat_extr.get_intermediate_layers(
                    x=x, n=layers, return_class_token=True
                )

            # Extract class token from the last layer and patch tokens from all extracted layers
            out_cls = intermediate[-1][1]  # (B, D)
            out_patch = tuple([p for (p, _) in intermediate])  # Tuple of (B, N, D)

        return out_cls, out_patch


class RADIO(nn.Module):
    """
    Wrapper for RADIO [https://arxiv.org/abs/2412.07679] feature extraction model.

    This wrapper extracts intermediate layer features for knowledge distillation purposes.

    Attributes:
        feat_extr (nn.Module): The loaded pretrained RADIO model

    Example:
        >>> model = RADIO()
        >>> summary, features = model(images)
    """

    def __init__(self) -> None:
        super().__init__()

        self.feat_extr = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-b",
            progress=False,
            skip_validation=True,
        )

        self.feat_extr.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract intermediate features from RADIO model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where:
                - B: batch size
                - C: number of channels (typically 3 for RGB)
                - H, W: image height and width

        Returns:
            Tuple containing:
                - summary (torch.Tensor): Global summary features from the model,
                                         shape (B, D) where D is embedding dimension
                - features (torch.Tensor): Patch-level features from intermediate layers
                                          shape depends on RADIO's internal architecture

        Note:
            Layer indices (8, 11) are selected to match similar semantic levels as
            DINOv2/v3 small/base models for consistency in distillation.
        """
        with torch.no_grad():
            # Extract features at layers 8 and 11 for compatibility with DINO models
            (summary, _), features = self.feat_extr.forward_intermediates(
                x, indices=LAYER_INDICES_SMALL_BASE
            )

        return summary, features
