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

def load_documents(path: str) -> list[tuple[Tensor, int]]:
    """
    Loads a set of patches and groups them into individual documents containing the
    grouped patch tokens and the writer ID
    
    Args:
        path (str): The location to load the patches from
    
    Returns:
        documents (list[tuple[Tensor, int]]): The documents in the specified patch file
    """
    
    data = torch.load(path)
    
    unique_documents, counts = torch.unique_consecutive(data['documents'], return_counts=True)
    grouped_patches: Tensor = torch.split(data['patches'], counts.tolist())
    
    return [(grouped_patches[i], unique_documents[i]) for i in range(len(unique_documents))]

def sample_document_patches(root: str, num_samples: int = 512, reset_device: bool = True, threads: int = 8) -> Tensor:
    """
    Collects a subsample of document patches to be used for VLAD codebook training
    
    This iteratively loads all patches from the `root` directory, groups them into documents, and takes a sample
    of it's patches. This is done to preserve memory while still getting a good sample of the documents
    """
    
    samples = []
    
    # load each patch file
    for path in sorted(Path(root).rglob("*")):
        documents = load_documents(path)
        
        for document, _ in documents:
            if reset_device:
                samples.append(document[torch.randperm(document.shape[0])[num_samples:]].to("cuda:0"))
            else:
                samples.append(document[torch.randperm(document.shape[0])[num_samples:]])

    return torch.cat(samples)