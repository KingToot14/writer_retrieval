from tqdm import tqdm

from writer_retrieval.data.serialization import load_documents
import writer_retrieval.models.vlad as vlad
import writer_retrieval.models.pca as pca

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
    
    # save VLAD
    codebook.save("output/models/historical_vits16/vlad.pt")
    
    # create document descriptors
    descriptors = []
    
    for document in tqdm(documents, desc="Creating VLAD Descriptors"):
        descriptors.append(codebook.create_descriptor(document))
    
    descriptors = torch.stack(descriptors)
    
    # train PCA
    pca_model = pca.PCAMatrix()
    pca_model.train(descriptors)
    
    # save PCA
    pca_model.save("output/models/historical_vits16/pca.model")