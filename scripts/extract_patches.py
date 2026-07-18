import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from writer_retrieval.data import TO_FLOAT, NORMALIZE
from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.data.serialization import save_patches
from writer_retrieval.models.dino import DINOModelv3
from writer_retrieval.data.filter import get_window_filter, get_patch_filter

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
    total_windows: int = 0
    filtered_windows: int = 0
    
    total_patches: int = 0
    filtered_patches: int = 0
    
    iteration = 0
    
    for batch in data:
        windows, writers, documents = batch
        
        # move to GPU and convert to float
        windows: torch.Tensor = windows.cuda(non_blocking=True)
        writers: torch.Tensor = writers.cuda(non_blocking=True)
        documents: torch.Tensor = documents.cuda(non_blocking=True)
        windows = TO_FLOAT(windows)
        
        # filter windows
        mask_win = get_window_filter(windows)
        win_count: int = mask_win.numel()
        
        # update tensors
        windows = windows[mask_win]
        writers = writers[mask_win]
        documents = documents[mask_win]
        
        # update stats
        total_windows += win_count
        filtered_windows += mask_win.sum().item()
        
        # get patch filter
        mask_patch = get_patch_filter(windows)
        
        # extract tokens
        windows = NORMALIZE(windows)
        tokens = model.extract_windows(windows)
        
        patch_count = mask_patch.numel()
        
        # filter patches
        window_ids, patch_ids = mask_patch.nonzero(as_tuple=True)
        
        tokens = tokens[window_ids, patch_ids]
        writers = writers[window_ids]
        documents = documents[window_ids]
        
        # update stats
        total_patches += patch_count
        filtered_patches += mask_patch.sum().item()
        
        data.set_postfix({
            "windows": f"{(filtered_windows / total_windows) * 100:.2f}%",
            "patches": f"{(filtered_patches / total_patches) * 100:.2f}%",
        })
        
        # store tokens
        save_patches(f"output/patches/patch_{iteration}.pt", tokens, writers, documents)
        iteration += 1