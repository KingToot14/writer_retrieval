import sys

from writer_retrieval.data.serialization import load_documents
import writer_retrieval.models.vlad as vlad

import torch

if __name__ == "__main__":
    documents = load_documents("output/patches/pretrained_vits16/train")
    target_samples = 512
    
    # collect subset for VLAD
    patches = []
    for document in documents:
        patch_count = document.shape[0]
        
        # collect random sample (if more than target samples)
        if patch_count <= target_samples or target_samples == -1:
            patches.append(document)
        else:
            patches.append(document[torch.randperm(patch_count)[:target_samples]])
        
    # train VLAD
    codebook = vlad.VLADCodebook()
    codebook.train(torch.cat(patches), niter=100)
    
    # create document descriptors
    test = codebook.create_descriptor(documents[0])
    
    print(test, test.shape)