import torch
from torch import Tensor

import faiss
import faiss.contrib.torch_utils

class VLADCodebook():
    def train(self, patches: Tensor, centroids: int = 100, niter: int = 25) -> None:
        """
        Trains a VLAD codebook using FAISS's KMeans
        
        Args:
            patches (Tensor): the patches to be used in training. This should be a subset of the full patches
            centroids (int): the number of centroids to use for the codebook (100 recommended)
            niter (int): the number of training iterations to perform
        """
        
        # create kmeans
        kmeans = faiss.Kmeans(
            d=patches.shape[1],
            k=centroids,
            niter=niter,
            verbose=True,
            gpu=True,
        )
        
        # trains kmeans
        kmeans.train(patches)
        
        self.kmeans = kmeans

    def create_descriptor(self, patches: Tensor) -> Tensor:
        """
        Creates an aggregated document descriptor based on the document foreground `patches` by running
        VLAD aggregation through the trained codebook
        """
        
        pass