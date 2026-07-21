import os
from pathlib import Path

import yaml
from types import SimpleNamespace
from argparse import ArgumentParser

import torch

from extract import extract_dataset
from train_models import train_vald_and_pca
from get_metrics import retrieve_writers

def dict_to_namespace(data: dict) -> SimpleNamespace:
    """
    A simple helper function that converts the `data` dictionary to a namespace that
    allows using dot notation for variable access
    """
    
    if isinstance(data, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    
    elif isinstance(data, list):
        return [dict_to_namespace(v) for v in data]
    
    return data

def main():
    # create parser
    parser = ArgumentParser()
    parser.add_argument("config_file")
    
    # parse arguments
    args = parser.parse_args()
    
    config: SimpleNamespace
    
    with open(args.config_file, "r") as fs:
        config = dict_to_namespace(yaml.load(fs, Loader=yaml.FullLoader))
    
    # set environment variables
    repo = Path(__file__).resolve().parents[1]
    
    # --- Patch Extraction --- #
    world_size = os.getenv('WORLD_SIZE')
    if not world_size:
        world_size = 1
    else:
        world_size = int(world_size)
    
    local_rank = os.getenv('LOCAL_RANK')
    if not local_rank:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    
    # parse arguments
    args = parser.parse_args()
    
    # print settings
    print(f"Extracting dataset: `{config.dataset}` ({config.run_name})")
    print(f"    Weights: `{config.weights}`")
    print(f"    Version: `{config.model.version}`")
    print(f"    Model:   `{config.model.kind}`")
    print(f"    Stride:  {config.extract.train_stride} (train), {config.extract.test_stride} (test)")
    
    # set PyTorch settings
    torch.backends.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"
    
    torch.set_float32_matmul_precision("medium")
    
    # extract training set
    extract_dataset(
        os.path.join(config.dataset, "train"),
        os.path.join(config.run_name, "train"),
        weight_path=config.weights,
        dino_version=config.model.version,
        use_dino_v1=config.model.kind == "dino_v1",
        max_windows=config.extract.num_windows,
        stride=config.extract.train_stride,
        world_size=world_size,
        local_rank=local_rank,
    )
    
    # extract testing set
    extract_dataset(
        os.path.join(config.dataset, "test"),
        os.path.join(config.run_name, "test"),
        weight_path=config.weights,
        dino_version=config.model.version,
        use_dino_v1=config.model.kind == "dino_v1",
        max_windows=config.extract.num_windows,
        stride=config.extract.test_stride,
        world_size=world_size,
        local_rank=local_rank,
    )
    
    # free non-main GPUs
    if local_rank != 0:
        return

    train_vald_and_pca(f"output/patches/{config.run_name}", config.run_name)
    retrieve_writers(f"output/patches/{config.run_name}", config.run_name)
    

if __name__ == "__main__":
    main()