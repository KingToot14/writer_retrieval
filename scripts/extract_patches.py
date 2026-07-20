import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from writer_retrieval.data import TO_FLOAT, NORMALIZE
from writer_retrieval.data.dataset import HistoricalWIDataset, WindowSampler, window_collate
from writer_retrieval.data.serialization import save_patches
from writer_retrieval.models.dino import *
from writer_retrieval.data.filter import get_window_filter, get_patch_filter

def extract_dataset(
        dataset_root: str,
        output_name: str,
        weight_path: str,
        dino_version: str,
        use_dino_v1: bool = False,
    ) -> None:
    
    """
    Extracts a collection of images using DINO
    """
    
    dataset = HistoricalWIDataset(dataset_root)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=WindowSampler(dataset, dataset.total_windows, 4096),
        collate_fn=window_collate,
        pin_memory=True,
    )
    
    # load dino model
    model: DINOModelBase
    
    if use_dino_v1:
        model = DINOModelv1(dino_version, weight_path, False)
    else:
        model = DINOModelv3(dino_version, weight_path, False)
    
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
        save_patches(f"output/patches/{output_name}/patch_{iteration}.pt", tokens, writers, documents)
        iteration += 1

if __name__ == "__main__":
    parser = ArgumentParser("Extract Patches")
    
    # add arguments
    parser.add_argument("dataset")
    parser.add_argument("run_name")
    parser.add_argument("-w", "--weights", default="weights/dinov3_vits16")
    parser.add_argument("-v", "--version", default="dinov3_vits16")
    parser.add_argument("--dino-v1", default=False, action="store_true")
    
    # parse arguments
    args = parser.parse_args()
    
    # print settings
    print(f"Extracting dataset: `{args.dataset}` ({args.run_name})")
    print(f"    Weights: `{args.weights}`")
    print(f"    Version: `{args.version}`")
    print(f"    Model:   `{"DINOv1" if args.dino_v1 else "DINOv3"}`")
    
    # extract training set
    extract_dataset(
        os.path.join(args.dataset, "train"),
        os.path.join(args.run_name, "train"),
        weight_path=args.weights,
        dino_version=args.version,
        use_dino_v1=args.dino_v1,
    )
    
    # extract testing set
    extract_dataset(
        os.path.join(args.dataset, "test"),
        os.path.join(args.run_name, "test"),
        weight_path=args.weights,
        dino_version=args.version,
        use_dino_v1=args.dino_v1,
    )