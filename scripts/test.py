from writer_retrieval.data import TO_FLOAT, NORMALIZE
from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.models.dino import DINOModelv3
from writer_retrieval.data.filter import get_window_filter

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
    
    # start processing
    data = tqdm(dataloader)
    total_filtered: int = 0
    
    for batch in data:
        windows, writers, documents = batch
        
        # move to GPU and convert to float
        windows: torch.Tensor = windows.cuda(non_blocking=True)
        writers: torch.Tensor = writers.cuda(non_blocking=True)
        documents: torch.Tensor = documents.cuda(non_blocking=True)
        windows = TO_FLOAT(windows)
        
        # filter windows
        mask = get_window_filter(windows)
        kept: int = mask.size()[0]
        
        # update tensors
        windows = windows[mask]
        writers = writers[mask]
        documents = documents[mask]
        
        # update stats
        total_filtered += windows.size()[0] - kept
        data.set_postfix({"filtered_windows": total_filtered})
        
        # extract tokens
        windows = NORMALIZE(windows)
        tokens = model.extract_windows(windows)
        
        