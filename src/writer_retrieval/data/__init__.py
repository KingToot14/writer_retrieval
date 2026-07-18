import torch
from torchvision.transforms import v2

# --- Constants --- #
EXTENSIONS = [".png", ".jpg", ".jpeg"]
WINDOW_SIZE = 224

TO_FLOAT = v2.ToDtype(torch.float32, scale=True).cuda()
NORMALIZE = v2.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
).cuda()

TRANSFORM = v2.Compose([TO_FLOAT, NORMALIZE]).cuda()