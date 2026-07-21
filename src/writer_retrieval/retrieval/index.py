import faiss
import faiss.contrib.torch_utils

import torch
from torch import Tensor

class WriterIndex:
    def __init__(self, dimensions: int = 384) -> None:
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatIP(
            self.res,
            dimensions,
        )
    
    def add(self, descriptors: Tensor) -> None:
        """
        Adds the `descriptors` to a new FAISS index
        """
        
        self.index.add(descriptors)
    
    def get_top_k(self, query: Tensor, k: int = 1) -> tuple[Tensor, Tensor]:
        """
        Returns the top `k` closest documents to the `query` document. This assumes that the most
        similar document will be itself, so `k` is increased by 1, and the 1st value is omitted
        """
        
        similarity, index = self.index.search(query.unsqueeze(0), k + 1)
        
        return similarity.squeeze()[1:], index.squeeze()[1:]