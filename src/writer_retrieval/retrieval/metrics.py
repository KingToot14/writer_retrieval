import torch
from torch import Tensor

import writer_retrieval.retrieval.index as retrieval_index

class Metrics:
    def __init__(self, descriptors: Tensor, writers: Tensor, index: retrieval_index.WriterIndex):
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
    
    def mean_average_precision(self, k: int = -1) -> float:
        """
        Calculates and returns the mean average precision. `k` is optional and specifies the maximum
        number of documents to consider "hits"
        """
        
        # get up to k matches (or all if k == -1)
        _, indices = self.index.get_top_k(self.descriptors, k)
        
        # get labels
        retrieved = self.writers[indices]
        queried = self.writers[:, None]
        
        # get hits over time
        relevant = retrieved == queried
        cumulative_hits = relevant.cumsum(dim=1)
        
        ranks = torch.arange(
            1, indices.shape[1] + 1,
            device=indices.device
        )
        
        # calculate average precision only at hits (where relevant is True)
        precision_at_rank = cumulative_hits / ranks
        ap = (precision_at_rank * relevant).sum(dim=1)
        
        # calculate the number of relevant documents per-query (minus self query)
        num_relevant = torch.tensor(
            [
                len(self.writer_map[self.writers[i].item()]) - 1
                for i in range(len(self.writers))
            ]
        )
        
        # calculate and return mAP
        ap /= num_relevant
        return ap.mean().item()