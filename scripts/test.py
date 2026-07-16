from writer_retrieval.data import TRANSFORM
from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.models.dino import DINOModelv3

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

if __name__ == "__main__":
    dataset = HistoricalWIDataset("datasets/historical_wi/test")
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=WindowSampler(dataset, dataset.total_windows, 4096),
        collate_fn=window_collate,
        pin_memory=True,
    )
    
    # load dino model
    model = DINOModelv3("dinov3_vits16", "weights/dinov3_vits16", False)
    
    for batch in tqdm(dataloader):
        windows, writers, documents = batch
        
        windows: torch.Tensor = windows.cuda(non_blocking=True)
        windows = TRANSFORM(windows)
        
        tokens = model.extract_windows(windows)