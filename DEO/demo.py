import torch

from DEO_weights import DEO

model = "swin_b"
path = "/home/filip/pretrained_weights/shortened/swin_bm_ms_fmow_hr_280k_100e_dinov3p34c_rgb_head_dconv_aug.pth"

model = DEO(model, path, "cuda")

image = torch.randn(1, 3, 256, 256, dtype=torch.float)
image = image.to("cuda")

features = model(image)
print([f.shape for f in features])
