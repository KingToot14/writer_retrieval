from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.models.dino import DINOModelv3

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from tqdm import tqdm

# --- Constants --- #
TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
]).cuda()

if __name__ == "__main__":
    dataset = HistoricalWIDataset("datasets/historical_wi/test")
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=WindowSampler(dataset, dataset.total_windows, 4096),
        collate_fn=window_collate,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    # load dino model
    model = DINOModelv3("dinov3_vits16", "weights/dinov3_vits16", False)
    
    for batch in tqdm(dataloader):
        windows, writers, documents = batch
        
        windows: torch.Tensor = windows.cuda(non_blocking=True)
        windows = TRANSFORM(windows)
        
        tokens = model.extract_windows(windows)