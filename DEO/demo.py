import torch

from DEO_weights import DEO

model = "swin_b"  # or vit_base
path = "/path/to/features/"

model = DEO(model, path, "cuda")

image = torch.randn(1, 3, 256, 256, dtype=torch.float)
image = image.to("cuda")

features = model(image)
print([f.shape for f in features])
