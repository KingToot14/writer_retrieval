import torch
from torch import Tensor

import writer_retrieval.retrieval.index as index

class Metrics:
    def __init__(self, descriptors: Tensor, writers: Tensor, index: index.WriterIndex):
        self.descriptors = descriptors
        self.writers = writers
        
        # create writer mapping
        self.writer_map: dict[int, Tensor] = {
            writer.item(): torch.where(writers == writer)[0]
            for writer in writers.unique()
        }
        
        self.index = index
    
    def top_k_accuracy(self, k: int) -> float:
        """
        Calculates and returns the top-k accuracy over all descriptors
        """
        
        _, indices = self.index.get_top_k(self.descriptors, k)
        
        retrieved = self.writers[indices]
        queried = self.writers.unsqueeze(1)
        
        correct = (retrieved == queried).any(dim=1)
        
        return correct.float().mean().item()