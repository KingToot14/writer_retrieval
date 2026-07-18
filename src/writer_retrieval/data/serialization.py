from pathlib import Path

import torch
from torch import Tensor

def save_patches(path: str, patches: Tensor, writers: Tensor, documents: Tensor) -> None:
    """
    Stores a set of patches
    
    Args:
        path (str): The location to store the patches at
        patches (Tensor): The patches extracted through DINO
        writers (Tensor): The writer IDs that map each patch to a writer
        documents (Tensor): The document IDs that map each patch to a specific document
    """
    
    # make sure folder exists
    Path("/".join(path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'patches': patches,
        'writers': writers,
        'documents': documents,
    }, path)

def load_patches(path: str) -> tuple[Tensor, Tensor, Tensor]:
    """
    Loads a set of patches containing 3 tensors: patch data, writer IDs, and document IDs
    
    Args:
        path (str): The location to load the patches from
    
    Returns:
        patches (Tensor): the patch data extracted from DINO
        writers (Tensor): the writer IDs for each of the extracted patches
        documents (Tensor): the document IDs for each of the extracted patches
    """
    
    data: dict[str, Tensor] = torch.load(path)
    
    return data['patches'], data['writers'], data['documents']