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

def load_patch(path: str) -> list[tuple[Tensor, int, int]]:
    """
    Loads a set of patches and groups them into individual documents containing the
    grouped patch tokens and the writer ID
    
    Args:
        path (str): The location to load the patches from
    
    Returns:
        documents (list[tuple[Tensor, int]]): The documents in the specified patch file
    """
    
    data = torch.load(path, map_location="cuda:0")
    
    unique_documents, counts = torch.unique_consecutive(data['documents'], return_counts=True)
    grouped_patches: Tensor = torch.split(data['patches'], counts.tolist())
    grouped_writers: Tensor = data['writers'][counts.cumsum(0) - counts[0]]
    
    return [
        (grouped_patches[i].contiguous(), int(grouped_writers[i]), int(unique_documents[i]))
        for i in range(len(unique_documents))]

def load_documents(root: str, target_samples: int = -1, threads: int = 8) -> Tensor:
    """
    Loads and groups all documents found in the patch files under the `root` directory. This is all done on
    the CPU for simplicity and performance. There's an optional `target_samples` parameter that can limit
    the number of patches collected per document (useful for VLAD codebook training).
    """
    
    paths = sorted(Path(root).rglob("*"))

    # helper function that acts as a target for thread workers
    def _load_and_sample(path: Path) -> list[Tensor]:
        sampled_documents = []
        
        for document, _, _ in load_patch(str(path)):
            patch_count = document.shape[0]
            
            # collect random sample (if more than target samples)
            if patch_count <= target_samples or target_samples == -1:
                sampled_documents.append(document)
            else:
                sampled_documents.append(document[torch.randperm(patch_count)[:target_samples]])
        
        return sampled_documents

    # start threaded work
    if threads <= 1:
        sampled_documents = [document for path in paths for document in _load_and_sample(path)]
    else:
        max_workers = max(1, min(threads, len(paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            sampled_documents = [document for batch in executor.map(_load_and_sample, paths) for document in batch]

    return sampled_documents