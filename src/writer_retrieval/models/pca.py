import faiss
import faiss.contrib.torch_utils

import torch
from torch import Tensor

class PCAMatrix:
    def train(self, descriptors: Tensor, target_dims: int = 384) -> None:
        """
        Trains a PCA model with whitening enabled. This reduces the dimensions of `descriptors` into
        `target_dims`
        """
        
        # create PCA
        self.pca = faiss.PCAMatrix(
            descriptors.shape[-1],
            target_dims,
            -0.5,           # whitening
        )
        
        self.pca.train(descriptors.cpu())
    
    def apply(self, descriptors: Tensor) -> Tensor:
        """
        Applies the trained PCA model to the given `descriptors` (must be batched)
        """
        
        return self.pca.apply_py(descriptors.cpu())