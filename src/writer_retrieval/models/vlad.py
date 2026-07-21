import torch
from torch import Tensor

import faiss
import faiss.contrib.torch_utils

class VLADCodebook():
    def train(self, patches: Tensor, k: int = 100, niter: int = 25) -> None:
        """
        Trains a VLAD codebook using FAISS's KMeans
        
        Args:
            patches (Tensor): the patches to be used in training. This should be a subset of the full patches
            centroids (int): the number of centroids to use for the codebook (100 recommended)
            niter (int): the number of training iterations to perform
        """
        
        self.k = k
        
        # create kmeans
        self.kmeans = faiss.Kmeans(
            d=patches.shape[1],
            k=k,
            niter=niter,
            verbose=True,
            gpu=True,
        )
        
        # trains kmeans
        self.kmeans.train(patches.cpu())
        self.centroids = torch.as_tensor(
            self.kmeans.centroids,
            device=patches.device,
            dtype=patches.dtype
        )
        
        # set index
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(
            self.res, patches.shape[1]
        )
        self.index.add(self.centroids)

    def create_descriptor(self, patches: Tensor) -> Tensor:
        """
        Creates an aggregated document descriptor based on the document foreground `patches` by running
        VLAD aggregation through the trained codebook
        """
        
        # calculate closest centroids
        _, indices = self.index.search(patches, 1)
        indices = indices.squeeze()
        residuals = patches - self.centroids[indices]
        
        # build descriptor
        descriptor = torch.zeros(self.k, patches.shape[1], device=patches.device)
        
        descriptor.index_add_(0, indices, residuals)
        descriptor = descriptor.reshape(-1)
        
        # TODO: normalize (if that's correct)
        
        return descriptor
        