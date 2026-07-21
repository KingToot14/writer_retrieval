from concurrent.futures import ThreadPoolExecutor
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
    
    data = torch.load(path, map_location="cpu")
    
    unique_documents, counts = torch.unique_consecutive(data['documents'], return_counts=True)
    grouped_patches: Tensor = torch.split(data['patches'], counts.tolist())
    
    return [(grouped_patches[i].contiguous(), int(unique_documents[i])) for i in range(len(unique_documents))]

def sample_document_patches(root: str, num_samples: int = 512, threads: int = 8) -> Tensor:
    """
    Collects a subsample of document patches to be used for VLAD codebook training.

    This iteratively loads all patches from the ``root`` directory, groups them into documents,
    and samples a fixed number of patches per document. The resulting Tensor is returned on CPU
    """
    
    paths = sorted(Path(root).rglob("*"))

    def _load_and_sample(path: Path) -> list[Tensor]:
        sampled_documents = []
        
        for document, _ in load_documents(str(path)):
            patch_count = document.shape[0]
            
            # collect random sample (if more than target samples)
            if patch_count <= num_samples:
                sampled_documents.append(document)
            else:
                sampled_documents.append(document[torch.randperm(patch_count)[:num_samples]])
        
        return sampled_documents

    if threads <= 1:
        sampled_documents = [document for path in paths for document in _load_and_sample(path)]
    else:
        max_workers = max(1, min(threads, len(paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            sampled_documents = [document for batch in executor.map(_load_and_sample, paths) for document in batch]

    return torch.cat(sampled_documents, dim=0)